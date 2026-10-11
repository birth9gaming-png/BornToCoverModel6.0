"""Private personal workspace persistence. Credentials never enter exports.

R2 writes use ETag preconditions to prevent an older browser from replacing a
newer save. Raw history objects are content addressed and uploaded only once.
"""
from __future__ import annotations
import copy
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime,timezone
from pathlib import Path

FORMAT='football-workspace-v1'
SESSION_KEYS=('cfb_historical_dataset','nfl_historical_dataset','cfb_training_result','nfl_training_result',
              'calibration_archive','calibration_report','calibration_signature')
HISTORY_KEYS=('cfb_historical_dataset','nfl_historical_dataset')
UI_KEYS=('controls_league','market_CFB','market_NFL','side_type_CFB','side_type_NFL','min_ev_CFB','min_ev_NFL',
         'training_validation','training_test','nfl_validation','nfl_test')
CREDENTIAL_KEYS={'apikey','apikeys','cfbdapikey','oddsapikey','password','secretaccesskey','accesskeyid','token',
                 'r2secretaccesskey','r2accesskeyid','r2accountid','r2endpointurl','r2bucket','cfbapilive','oddsapilive','historykey',
                 'secret','secrets','credentials','authorization','apitoken','cfbdkey','oddskey'}


class StorageError(RuntimeError):pass
class SaveConflict(StorageError):pass


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items() if re.sub(r'[^a-z0-9]','',str(k).lower()) not in CREDENTIAL_KEYS}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if hasattr(value,'item'):return clean(value.item())
    return value


def encode(value):return json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def fingerprint(value):return hashlib.sha256(encode(value)).hexdigest()


def make_bundle(data,session):
    return clean(dict(format=FORMAT,snapshot=data,session={k:session[k] for k in SESSION_KEYS if k in session},
                      ui={k:session[k] for k in UI_KEYS if k in session}))


def validate_bundle(payload):
    from model import validate,ensure_cfb_defaults
    from nfl_layers import settings as nfl_settings
    from auto_injuries import settings as injury_settings
    from app_lifecycle import preferences
    if not isinstance(payload,dict) or payload.get('format')!=FORMAT:raise ValueError('Use a complete football workspace backup.')
    data=copy.deepcopy(payload.get('snapshot'));validate(data);ensure_cfb_defaults(data)
    nfl_settings(data);injury_settings(data);preferences(data)
    state=payload.get('session',{});ui=payload.get('ui',{})
    if not isinstance(state,dict) or not isinstance(ui,dict):raise ValueError('Invalid saved workspace state.')
    for key,fmt in [('cfb_historical_dataset','cfbd-history-v1'),('nfl_historical_dataset','nfl-history-v1')]:
        if key in state:
            history=state[key]
            if not isinstance(history,dict) or history.get('format')!=fmt:raise ValueError('Invalid checked history: '+key)
            required=('games','box','advanced','lines','files','issues') if key.startswith('cfb') else ('games','team_games','files','issues')
            if any(not isinstance(history.get(k),list) for k in required):raise ValueError('Incomplete checked history: '+key)
    for lg in ('cfb','nfl'):
        if data.get(lg+'_boost_model'):
            if lg=='cfb':
                from cfb_ml import correction,FEATURES
                correction(data[lg+'_boost_model'],dict.fromkeys(FEATURES,0.))
            else:
                from nfl_ml import correction
                correction(data[lg+'_boost_model'],{})
        result=state.get(lg+'_training_result')
        if result is not None:
            if not isinstance(result,dict) or not isinstance(result.get('model'),dict) or not isinstance(result.get('report'),dict):raise ValueError('Incomplete saved training result: '+lg)
            report=result['report']
            if not isinstance(report.get('targets'),dict) or not isinstance(report.get('counts'),dict):raise ValueError('Invalid saved fit report: '+lg)
            for target in ('margin','total'):
                sigma=float(report['targets'][target]['validation_sigma'])
                if not 0<sigma<100:raise ValueError('Invalid saved validation uncertainty.')
            result.setdefault('forecasts',{})
            if lg=='cfb':
                from cfb_ml import correction,FEATURES
                correction(result['model'],dict.fromkeys(FEATURES,0.))
            else:
                from nfl_ml import correction
                correction(result['model'],{})
    encode(payload)
    checked_ui={k:copy.deepcopy(ui[k]) for k in UI_KEYS if k in ui}
    for key in ('controls_league',):
        if key in checked_ui and checked_ui[key] not in ('CFB','NFL'):raise ValueError('Invalid saved league selection.')
    for lg in ('CFB','NFL'):
        for stem,choices in [('market_',('All','Spreads','Totals')),('side_type_',('All','Favorites','Underdogs'))]:
            if stem+lg in checked_ui and checked_ui[stem+lg] not in choices:raise ValueError('Invalid saved market filter.')
        key='min_ev_'+lg
        if key in checked_ui and (isinstance(checked_ui[key],bool) or not isinstance(checked_ui[key],(int,float)) or not 0<=checked_ui[key]<=100):raise ValueError('Invalid saved EV filter.')
    for key,minimum in [('training_validation',2005),('training_test',2006),('nfl_validation',2000),('nfl_test',2001)]:
        if key in checked_ui and (isinstance(checked_ui[key],bool) or not isinstance(checked_ui[key],int) or not minimum<=checked_ui[key]<=2100):raise ValueError('Invalid saved training split.')
    return clean(dict(format=FORMAT,snapshot=data,session={k:copy.deepcopy(state[k]) for k in SESSION_KEYS if k in state},ui=checked_ui))


def restore_session(payload,session):
    checked=validate_bundle(payload)
    for key in SESSION_KEYS:
        if key in checked['session']:session[key]=checked['session'][key]
        else:session.pop(key,None)
    for key,value in checked['ui'].items():session[key]=value
    return checked['snapshot']


class R2Store:
    def __init__(self,bucket,client,prefix='football-model'):
        self.bucket=bucket;self.client=client;self.prefix=prefix.strip('/');self.last_warning=None;self._history_objects=set()

    def read(self,key):
        try:
            result=self.client.get_object(Bucket=self.bucket,Key=self.prefix+'/'+key)
            return result['Body'].read(),result.get('ETag'),result.get('Metadata',{})
        except Exception as exc:
            code=(getattr(exc,'response',{}).get('Error') or {}).get('Code','')
            if code in ('NoSuchKey','404','NotFound'):return None,None,{}
            raise StorageError('Cloud read failed'+(' ('+str(code)+')' if code else '')+'. Check bucket permissions and saved credentials.') from None

    def write(self,key,content,expected=None,immutable=False,metadata=None):
        args=dict(Bucket=self.bucket,Key=self.prefix+'/'+key,Body=content,ContentType='application/json',
                  Metadata=metadata or {},CacheControl='private, no-store')
        if expected is not None:args['IfMatch']=expected
        else:args['IfNoneMatch']='*'
        try:return self.client.put_object(**args).get('ETag')
        except Exception as exc:
            code=(getattr(exc,'response',{}).get('Error') or {}).get('Code','')
            if code in ('PreconditionFailed','ConditionalRequestConflict','412','409'):
                if immutable:return None
                raise SaveConflict('A newer cloud save exists. Reload that save before changing it; download this session as a backup first.') from None
            raise StorageError('Cloud save failed'+(' ('+str(code)+')' if code else '')+'. Your session remains available; download a complete backup.') from None

    def load(self):
        raw,etag,_=self.read('latest.json.gz')
        if raw is None:return None,None
        try:
            payload=json.loads(gzip.decompress(raw))
            for key,ref in payload.pop('history_refs',{}).items():
                if key not in HISTORY_KEYS or not re.fullmatch(r'[a-f0-9]{64}',str(ref)):raise ValueError('Invalid saved history reference.')
                content,_,_=self.read('history/'+ref+'.json.gz')
                if content is None:raise ValueError('Saved history object is missing.')
                history=json.loads(gzip.decompress(content))
                if fingerprint(history)!=ref:raise ValueError('Saved history checksum does not match.')
                self._history_objects.add(ref)
                payload.setdefault('session',{})[key]=history
            return validate_bundle(payload),etag
        except (ValueError,TypeError,KeyError,OSError,EOFError):raise StorageError('Cloud workspace is incomplete or invalid. Automatic saving is paused to protect the saved copy.') from None

    def save(self,bundle,expected=None):
        payload=copy.deepcopy(bundle);payload['history_refs']={}
        for key in HISTORY_KEYS:
            history=payload['session'].pop(key,None)
            if history is not None:
                digest=fingerprint(history)
                if digest not in self._history_objects:
                    self.write('history/'+digest+'.json.gz',gzip.compress(encode(history),mtime=0),immutable=True)
                    self._history_objects.add(digest)
                payload['history_refs'][key]=digest
        payload['saved_at']=datetime.now(timezone.utc).isoformat()
        # Keep the previous compact manifest; immutable histories already remain.
        if expected:
            previous,old_etag,_=self.read('latest.json.gz')
            if old_etag!=expected:raise SaveConflict('A newer cloud save exists. Download this session, then reload the cloud save.')
            if previous:
                _,backup_etag,_=self.read('previous.json.gz')
                self.write('previous.json.gz',previous,expected=backup_etag)
        return self.write('latest.json.gz',gzip.compress(encode(payload),mtime=0),expected=expected)

    def cache_get(self,key,ttl):
        try:
            content,_,meta=self.read('cache/'+key+'.gz')
            age=time.time()-float(meta.get('fetched-at',0))
            return gzip.decompress(content) if content is not None and 0<=age<ttl else None
        except (StorageError,ValueError,OSError,EOFError):return None

    def cache_put(self,key,content):
        if len(content)>100*1024*1024:return
        try:
            _,etag,_=self.read('cache/'+key+'.gz')
            self.write('cache/'+key+'.gz',gzip.compress(content,compresslevel=1,mtime=0),expected=etag,metadata={'fetched-at':str(time.time())})
        except StorageError:self.last_warning='Some downloaded data could not be cached in cloud storage.'


class LocalStore:
    """For an explicitly configured durable directory, e.g. a Railway volume."""
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.last_warning=None
    def load(self):
        file=self.path/'workspace.json.gz'
        if not file.exists():return None,None
        raw=file.read_bytes()
        try:return validate_bundle(json.loads(gzip.decompress(raw))),hashlib.sha256(raw).hexdigest()
        except (ValueError,TypeError,OSError,EOFError):raise StorageError('Saved workspace is invalid; automatic saving is paused.') from None
    def save(self,bundle,expected=None):
        # Inter-process lock plus atomic replacement on hosted Linux or local Windows.
        with (self.path/'workspace.lock').open('a+b') as lock:
            if os.name=='nt':
                import msvcrt
                if lock.tell()==0:lock.write(b'0');lock.flush()
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
            else:
                import fcntl
                fcntl.flock(lock,fcntl.LOCK_EX)
            file=self.path/'workspace.json.gz'
            current=hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
            if current!=expected:raise SaveConflict('A newer saved workspace exists; reload it before saving.')
            payload={**bundle,'saved_at':datetime.now(timezone.utc).isoformat()};raw=gzip.compress(encode(payload),mtime=0)
            if file.exists():(self.path/'previous.json.gz').write_bytes(file.read_bytes())
            fd,temp=tempfile.mkstemp(dir=self.path)
            with os.fdopen(fd,'wb') as target:target.write(raw)
            os.replace(temp,file)
            return hashlib.sha256(raw).hexdigest()
    def cache_get(self,key,ttl):
        file=self.path/'cache'/(key+'.bin')
        if file.exists() and 0<=time.time()-file.stat().st_mtime<ttl:return file.read_bytes()
        return None
    def cache_put(self,key,content):
        folder=self.path/'cache';folder.mkdir(exist_ok=True)
        fd,temp=tempfile.mkstemp(dir=folder)
        with os.fdopen(fd,'wb') as target:target.write(content)
        os.replace(temp,folder/(key+'.bin'))


def configured_store(secret):
    directory=str(secret('MODEL_DATA_DIR') or os.getenv('RAILWAY_VOLUME_MOUNT_PATH','')).strip()
    if directory:return LocalStore(directory),'Configured data directory'
    names=('R2_ACCOUNT_ID','R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY','R2_BUCKET')
    values={k:str(secret(k) or '').strip() for k in names}
    if not any(values.values()):return None,'Not connected'
    if not all(values.values()):raise StorageError('R2 setup is incomplete. Save all four R2 settings in host Secrets.')
    if not re.fullmatch(r'[a-fA-F0-9]{32}',values['R2_ACCOUNT_ID']):raise StorageError('R2_ACCOUNT_ID must be the 32-character account ID.')
    import boto3
    from botocore.config import Config
    client=boto3.client('s3',endpoint_url='https://'+values['R2_ACCOUNT_ID']+'.r2.cloudflarestorage.com',
        aws_access_key_id=values['R2_ACCESS_KEY_ID'],aws_secret_access_key=values['R2_SECRET_ACCESS_KEY'],region_name='auto',
        config=Config(connect_timeout=5,read_timeout=15,retries={'max_attempts':1},
                      request_checksum_calculation='when_required',response_checksum_validation='when_required'))
    return R2Store(values['R2_BUCKET'],client),'Cloudflare R2'
