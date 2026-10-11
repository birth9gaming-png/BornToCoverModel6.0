"""Streamlit session integration for automatic private save/restore."""
from __future__ import annotations
import copy
import json
import time
from datetime import datetime, timezone,timedelta
from pathlib import Path
import streamlit as st
from persistence import configured_store,make_bundle,fingerprint,restore_session,StorageError,SaveConflict
from data_pipeline import set_cache_store
from background_work import FeedJob,report_progress,RefreshCancelled


def _slate_source(data):
    return fingerprint([data.get('mode'),data.get('as_of'),data.get('season'),data.get('window'),
                        [str(g.get('id')) for g in data.get('games',[])]])


def _injuries_due(data,now):
    from auto_injuries import settings
    from matchup_history import timestamp
    config=settings(data)
    if not config['enabled'] or data.get('mode')!='connected':return False
    for league in ('NFL','CFB'):
        if not config[league]:continue
        try:age=(now-timestamp(data.get('auto_injury_checks',{}).get(league,{}).get('checked_at'))).total_seconds()
        except ValueError:age=float('inf')
        if 0<=age<config['refresh_minutes']*60:continue
        for game in data.get('games',[]):
            try:
                if game.get('league')==league and now<timestamp(game.get('kickoff'))<=now+timedelta(days=8):return True
            except ValueError:pass
    return False


def _merge_current_settings(current,updated):
    from app_lifecycle import PRESERVE
    from cfb_metrics import preserve_cfb_overrides
    out=preserve_cfb_overrides(current,updated)
    out['parameters']=copy.deepcopy(current['parameters'])
    for key in PRESERVE+('nfl_layer_settings','odds_alias_overrides'):
        if key not in ('auto_injury_reports','auto_injury_checks') and key in current:
            out[key]=copy.deepcopy(current[key])
    if current.get('nfl_provider_overrides'):
        from nfl_inputs import merge_inputs
        out=merge_inputs(out,current['nfl_provider_overrides'])
        out['nfl_provider_overrides']=copy.deepcopy(current['nfl_provider_overrides'])
    return out


def startup_refresh(data,secret,now=None):
    """Start/poll detached downloads; return immediately to render all controls."""
    from app_lifecycle import automatic_refresh,refresh_due
    from auto_injuries import refresh as refresh_injuries
    now=now or datetime.now(timezone.utc)
    job=st.session_state.get('_feed_job')
    if job:
        if not job.future.done():return data,False
        st.session_state.pop('_feed_job',None)
        try:
            updated,changed=job.future.result()
            if changed and job.source==_slate_source(data):
                st.session_state.pop('_refresh_error',None)
                return _merge_current_settings(data,updated),True
        except RefreshCancelled as exc:st.session_state['_refresh_error']=str(exc)
        except Exception as exc:
            st.session_state['_refresh_error']='Automatic update retained your saved inputs. '+str(exc)
    attempt=st.session_state.get('_auto_attempt_at')
    if attempt is not None and (now-attempt).total_seconds()<300:return data,False
    keys={name:secret(name) for name in ('CFBD_API_KEY','ODDS_API_KEY')}
    needs_slate=refresh_due(data,now) and (data.get('mode')=='connected' or any(keys.values()))
    if not needs_slate and not _injuries_due(data,now):return data,False
    incoming=copy.deepcopy(data)
    def work():
        report_progress('Loading upcoming schedules, statistics and odds')
        out,changed=automatic_refresh(incoming,lambda name:keys.get(name,''),now)
        if out.get('mode')=='connected':
            report_progress('Checking current injury availability')
            changed=refresh_injuries(out,now=now) or changed
        return out,changed
    st.session_state['_auto_attempt_at']=now
    st.session_state['_feed_job']=FeedJob(work,_slate_source(data))
    return data,False


def start_manual_refresh(data,season,start,end,leagues,cfbd_key='',odds_key='',regions='us'):
    from app_lifecycle import refresh_snapshot
    from auto_injuries import refresh as refresh_injuries
    old=st.session_state.get('_feed_job')
    if old:old.cancel()
    incoming=copy.deepcopy(data)
    def work():
        report_progress('Loading the requested slate')
        out=refresh_snapshot(incoming,season,start,end,leagues,cfbd_key,odds_key,regions)
        report_progress('Checking current injury availability')
        refresh_injuries(out,force=True)
        return out,True
    st.session_state['_auto_attempt_at']=datetime.now(timezone.utc)
    st.session_state['_feed_job']=FeedJob(work,_slate_source(data),kind='manual')


def bootstrap(secret,root,apply_initial_profile):
    pending=st.session_state.pop('_pending_workspace',None)
    if pending:
        for key in list(st.session_state):
            if key.startswith(('weight_','scenario_','team_','selected_matchup_','nfl_layer_','auto_injury_','app_pref_')) or key in ('controls_league','market_CFB','market_NFL','side_type_CFB','side_type_NFL','min_ev_CFB','min_ev_NFL','training_validation','training_test','nfl_validation','nfl_test','dataset_json_editor'):
                st.session_state.pop(key,None)
        data=restore_session(pending['payload'],st.session_state)
        st.session_state['data']=data;st.session_state['baseline']=copy.deepcopy(data)
        if pending.get('cloud'):
            st.session_state['_store_etag']=pending['etag'];st.session_state['_save_allowed']=True
            st.session_state['_saved_digest']=fingerprint(make_bundle(data,st.session_state))
            st.session_state.pop('_storage_error',None)
    if '_storage_initialized' not in st.session_state:
        st.session_state['_storage_initialized']=True
        st.session_state['_save_allowed']=True
        try:
            store,label=configured_store(secret)
            st.session_state['_store']=store;st.session_state['_store_label']=label
            set_cache_store(store)
            if store:
                with st.spinner('Loading your saved workspace…'):
                    payload,etag=store.load()
                st.session_state['_store_etag']=etag
                if payload:
                    data=restore_session(payload,st.session_state)
                    st.session_state['data']=data;st.session_state['baseline']=copy.deepcopy(data)
                    st.session_state['_saved_digest']=fingerprint(make_bundle(data,st.session_state))
                    st.session_state['_storage_message']='Saved workspace restored automatically.'
        except (StorageError,ValueError,TypeError,OSError) as exc:
            st.session_state['_storage_error']=str(exc);st.session_state['_save_allowed']=False
    set_cache_store(st.session_state.get('_store'))
    if 'data' not in st.session_state:
        from model import ensure_cfb_defaults,validate
        data=ensure_cfb_defaults(json.loads((Path(root)/'snapshot.json').read_text()));apply_initial_profile(data);validate(data)
        st.session_state['data']=data;st.session_state['baseline']=copy.deepcopy(data)
        # Compact checked NFL history is bundled; no multi-season download is needed.
        included=Path(root)/'nfl_history_2016_2025.json'
        if included.exists():st.session_state.setdefault('nfl_historical_dataset',json.loads(included.read_text()))
        fit=Path(root)/'nfl_reference_fit.json'
        if fit.exists():
            from nfl_ml import signature
            payload=json.loads(fit.read_text())
            if payload['model']['core_signature']==signature(data['parameters']):
                st.session_state.setdefault('nfl_training_result',{**payload,'forecasts':{}})
                data['nfl_boost_model']=payload['model'];data['nfl_training_report']=payload['report']
                for target,key in [('margin','sigma_spread'),('total','sigma_total')]:
                    data['parameters']['NFL'][key]=payload['report']['targets'][target]['validation_sigma']
                st.session_state['baseline']=copy.deepcopy(data)
    from model import ensure_cfb_defaults
    from nfl_layers import settings as nfl_settings
    from auto_injuries import settings as injury_settings
    from app_lifecycle import preferences
    data=st.session_state['data'];ensure_cfb_defaults(data)
    data['nfl_layer_settings']=nfl_settings(data)
    data['auto_injury_settings']=injury_settings(data)
    data['app_preferences']=preferences(data)


def autosave(force=False):
    store=st.session_state.get('_store')
    if not store or not st.session_state.get('_save_allowed') or 'data' not in st.session_state:return False
    try:
        bundle=make_bundle(st.session_state['data'],st.session_state);digest=fingerprint(bundle)
        if not force and digest==st.session_state.get('_saved_digest'):return True
        etag=store.save(bundle,st.session_state.get('_store_etag'))
        st.session_state['_store_etag']=etag;st.session_state['_saved_digest']=digest
        st.session_state['_saved_at']=datetime.now(timezone.utc).isoformat(timespec='seconds')
        st.session_state['_storage_message']='Saved automatically.';st.session_state.pop('_storage_error',None)
        return True
    except SaveConflict as exc:
        st.session_state['_storage_error']=str(exc);st.session_state['_save_allowed']=False
    except (StorageError,ValueError,TypeError,OSError) as exc:
        st.session_state['_storage_error']=str(exc)
    return False


def rerun():
    if '_pending_workspace' not in st.session_state:autosave()
    st.rerun()


def load_saved(reset_widgets):
    store=st.session_state.get('_store')
    if not store:raise StorageError('Add persistent-storage settings in host Secrets first.')
    payload,etag=store.load()
    if payload is None:raise StorageError('No saved workspace was found yet.')
    st.session_state['_pending_workspace']=dict(payload=payload,cloud=True,etag=etag)


def periodic_checks():
    """Refresh while the browser is open; sleeping hosts resume on next visit."""
    active=st.session_state.get('_feed_job')
    @st.fragment(run_every='2s' if active else '5m')
    def timer():
        job=st.session_state.get('_feed_job')
        if job:
            if job.future.done():st.rerun(scope='app')
            message,seconds=job.status()
            st.info('Updating data in the background. Setup and your saved inputs remain available.')
            st.caption(message+f' · {seconds//60}:{seconds%60:02d} elapsed')
        # Only a timer-driven fragment rerun starts a full app refresh.
        last=st.session_state.get('_timer_tick',0.)
        if last and time.monotonic()-last>=285:
            st.session_state['_timer_tick']=time.monotonic()
            st.rerun(scope='app')
        st.session_state.setdefault('_timer_tick',time.monotonic())
    timer()
