"""Free NFL season downloads, compact imports, and pre-week training rows."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
import zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import pandas as pd
from audit_time import observed_time
from nfl_stats import PBP_COLUMNS,code,game_summaries,team_features

FORMAT='nfl-history-v1'
MAX_BYTES=200_000_000
STAT_FIELDS=('plays','yards','epa_sum','successes','pass_plays','explosive_passes','dropbacks',
    'pass_epa_sum','rush_plays','rush_epa_sum','sacks','sack_observations','cpoe_sum','cpoe_count',
    'invalid_core_plays','net_turnovers')


def finite(value):
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def normalize_game(row):
    raw='game_id' in row
    game=dict(id=str(row['game_id'] if raw else row['id']),season=int(row['season']),week=int(row['week']),
        away=code(row['away_team'] if raw else row['away']),home=code(row['home_team'] if raw else row['home']),
        season_type=('regular' if str(row.get('game_type','REG')).upper()=='REG' else 'postseason') if raw else row.get('season_type','regular'))
    clock=row.get('gametime')
    value=str(row.get('gameday'))+' '+str(clock)+' America/New_York' if raw else row.get('kickoff')
    game['kickoff']=observed_time(value).isoformat()
    if raw:
        venue=str(row.get('location','')).lower()
        game['neutral']=True if venue=='neutral' else False if venue=='home' else None
    else:game['neutral']=row.get('neutral') if isinstance(row.get('neutral'),bool) else None
    for field in ('away_score','home_score','away_rest','home_rest'):game[field]=finite(row.get(field))
    game['line_away']=finite(row.get('spread_line') if raw else row.get('line_away'))
    game['market_total']=finite(row.get('total_line') if raw else row.get('market_total'))
    scores=(game['away_score'],game['home_score'])
    game['completed']=all(v is not None and v>=0 and v==int(v) for v in scores) and row.get('completed',True) is not False
    if game['away']==game['home'] or not 1999<=game['season']<=2100 or not 1<=game['week']<=30:raise ValueError('Invalid NFL season, week or team pairing.')
    if game['season_type'] not in ('regular','postseason'):raise ValueError('Use regular-season or postseason NFL games.')
    return game


def normalize_stats(row):
    result=dict(id=str(row['id']),team=code(row['team']))
    for key in STAT_FIELDS:
        value=finite(row.get(key))
        if value is None:raise ValueError('Missing NFL additive statistic: '+key)
        result[key]=value
    for key in ('plays','successes','pass_plays','explosive_passes','dropbacks','rush_plays','sacks','sack_observations','cpoe_count','invalid_core_plays'):
        if result[key]<0 or result[key]!=int(result[key]):raise ValueError('Invalid NFL count: '+key)
    if result['plays']<=0 or result['successes']>result['plays'] or result['pass_plays']>result['plays'] or result['explosive_passes']>result['pass_plays']:
        raise ValueError('NFL play counts are inconsistent.')
    return result


def blank():
    return dict(format=FORMAT,league='NFL',created_at=datetime.now(timezone.utc).isoformat(),games=[],team_games=[],files=[],issues=[])


def read_pbp(content,filename):
    compression='gzip' if filename.lower().endswith('.gz') else None
    frame=pd.read_csv(io.BytesIO(content),compression=compression,usecols=lambda c:c in PBP_COLUMNS,low_memory=False)
    return game_summaries(frame)


def import_files(files,existing=None):
    maps={'games':{},'team_games':{}};conflicts={key:set() for key in maps};issues=Counter();reports=[]
    def keep(category,value):
        key=value['id'] if category=='games' else (value['id'],value['team'])
        if key in conflicts[category]:return
        if key in maps[category] and maps[category][key]!=value:
            maps[category].pop(key);conflicts[category].add(key);issues[category+': conflicting duplicates withheld']+=1
        else:maps[category][key]=value
    if existing:
        if existing.get('format')!=FORMAT:raise ValueError('Use an NFL historical dataset.')
        for row in existing['games']:keep('games',normalize_game(row))
        for row in existing['team_games']:keep('team_games',normalize_stats(row))
    pending=[];total=0
    for name,content in files:
        total+=len(content)
        if total>MAX_BYTES:raise ValueError('Split history uploads into batches smaller than 200 MB.')
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                if len(z.infolist())>2000:raise ValueError('ZIP has too many members.')
                for member in z.infolist():
                    if member.is_dir() or member.filename.startswith('__MACOSX/'):continue
                    if not member.filename.lower().endswith(('.csv','.csv.gz','.json')):continue
                    total+=member.file_size
                    if total>MAX_BYTES:raise ValueError('ZIP exceeds 200 MB when expanded; split your seasons.')
                    pending.append((member.filename,z.read(member)))
        else:pending.append((name,content))
    for name,content in pending:
        report=dict(file=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),games=0,team_games=0,rejected=0)
        try:
            if name.lower().endswith('.json'):
                payload=json.loads(content)
                if not isinstance(payload,dict) or payload.get('format')!=FORMAT:raise ValueError('JSON must be an exported nfl-history-v1 dataset.')
                categories=[('games',payload['games']),('team_games',payload['team_games'])]
            else:
                frame=pd.read_csv(io.BytesIO(content),compression='gzip' if name.lower().endswith('.gz') else None,usecols=lambda c:c in PBP_COLUMNS or c in {'season','game_type','away_team','home_team','gameday','gametime','location','away_score','home_score','away_rest','home_rest','spread_line','total_line'} or c in STAT_FIELDS or c in {'id','team'},low_memory=False)
                if 'posteam' in frame:categories=[('team_games',game_summaries(frame))]
                elif 'away_team' in frame:
                    if 'game_type' not in frame:raise ValueError('NFL schedule CSV needs game_type to exclude preseason.')
                    categories=[('games',frame[frame['game_type'].isin(['REG','WC','DIV','CON','SB'])].to_dict('records'))]
                elif {'id','team','plays'}<=set(frame):categories=[('team_games',frame.to_dict('records'))]
                else:raise ValueError('Use nflverse games.csv and play_by_play_YEAR.csv.gz, or an exported NFL dataset. Season averages are not game-level data.')
            for category,rows in categories:
                for row in rows:
                    try:
                        value=normalize_game(row) if category=='games' else normalize_stats(row)
                        keep(category,value);report[category]+=1
                    except (ValueError,KeyError,TypeError) as exc:report['rejected']+=1;issues[str(exc)]+=1
        except (ValueError,KeyError,TypeError,UnicodeError) as exc:report['rejected']+=1;issues[str(exc)]+=1
        reports.append(report)
    result=blank()
    result.update(games=sorted(maps['games'].values(),key=lambda g:(g['season'],g['week'],g['id'])),
        team_games=sorted(maps['team_games'].values(),key=lambda r:(r['id'],r['team'])),files=reports,
        issues=[dict(reason=k,count=v) for k,v in issues.items()])
    return result


def coverage(dataset):
    stats={(r['id'],r['team']):r for r in dataset['team_games']};output={}
    for game in dataset['games']:
        row=output.setdefault(game['season'],dict(season=game['season'],games=0,completed=0,both_team_stats=0,with_spread=0,with_total=0,unknown_venue=0))
        row['games']+=1;row['completed']+=game['completed'];row['unknown_venue']+=game['neutral'] is None
        row['both_team_stats']+=all((game['id'],game[s]) in stats and stats[(game['id'],game[s])]['invalid_core_plays']==0 for s in ('away','home'))
        row['with_spread']+=game['line_away'] is not None;row['with_total']+=game['market_total'] is not None
    return [output[k] for k in sorted(output)]


def fetch_years(years,fetch=None,progress=None,existing=None,include_postseason=False):
    from data_pipeline import NFL_SCHEDULE,NFL_PBP,request_bytes
    fetch=fetch or request_bytes
    years=sorted({int(y) for y in years})
    if not years or len(years)>20 or min(years)<1999 or max(years)>datetime.now().year:raise ValueError('Select 1–20 NFL seasons from 1999 through this year.')
    frame=pd.read_csv(io.BytesIO(fetch(NFL_SCHEDULE,ttl=3600)));dataset=existing or blank();errors=[]
    types=['REG','WC','DIV','CON','SB'] if include_postseason else ['REG']
    schedule=frame[frame['season'].isin(years) & frame['game_type'].isin(types)]
    dataset=import_files([('nfl_selected_games.csv',schedule.to_csv(index=False).encode())],dataset)
    for i,year in enumerate(years):
        if progress:progress(i/len(years),f'Downloading {year} ({i+1}/{len(years)})')
        try:
            content=fetch(NFL_PBP.format(season=year),ttl=365*86400 if year<datetime.now().year else 3600)
            summaries=read_pbp(content,f'play_by_play_{year}.csv.gz')
            wanted={g['id'] for g in dataset['games'] if g['season']==year}
            compact=blank();compact['team_games']=[r for r in summaries if r['id'] in wanted]
            dataset=import_files([(f'nfl_{year}_compact.json',json.dumps(compact,allow_nan=False).encode())],dataset)
            del content,summaries
        except (RuntimeError,ValueError,KeyError,TypeError) as exc:errors.append(f'{year}: '+str(exc))
        if progress:progress((i+1)/len(years),f'Processed {year}; compact game summaries retained')
    return dataset,errors


def dataset_hash(dataset):
    return hashlib.sha256(json.dumps({k:dataset[k] for k in ('games','team_games')},sort_keys=True,allow_nan=False).encode()).hexdigest()


def build_rows(dataset,parameters,min_games=2,progress=None):
    from model import ensure_cfb_defaults,project
    from nfl_ml import features
    from data_pipeline import NFL_NAMES
    params=copy.deepcopy(parameters)
    for k in ('margin_scale','total_scale'):params['NFL'][k]=1.
    for k in ('margin_intercept','total_intercept'):params['NFL'][k]=0.
    now=datetime.now(timezone.utc)
    completed=[g for g in dataset['games'] if g['completed'] and g['neutral'] is not None and observed_time(g['kickoff'])+timedelta(hours=12)<now]
    groups=defaultdict(list)
    for g in completed:groups[(g['season'],g['week'])].append(g)
    rows=[];skipped=Counter()
    for i,((season,week),targets) in enumerate(sorted(groups.items())):
        cutoff=min(observed_time(g['kickoff']) for g in dataset['games'] if (g['season'],g['week'])==(season,week))
        prior=[g for g in completed if g['season']==season and g['week']<week and observed_time(g['kickoff'])+timedelta(hours=12)<cutoff]
        selected={g['id'] for g in prior}
        stats=[r for r in dataset['team_games'] if r['id'] in selected]
        teams=team_features(stats,prior,NFL_NAMES,week-1)
        teams={k:v for k,v in teams.items() if v['sample_games']>=min_games}
        history=[dict(g,league='NFL',away='NFL:'+g['away'],home='NFL:'+g['home']) for g in prior]
        data=dict(parameters=params,teams=teams,games=[],season=season,rating_history=history,history=[],schedule=[])
        ensure_cfb_defaults(data)
        for target in targets:
            game={**target,'league':'NFL','away':'NFL:'+target['away'],'home':'NFL:'+target['home'],
                  'feature_through_week':week-1,'line_away':None,'total':None,'quotes':[]}
            try:
                result=project(data,game)
                if 'away' not in result:skipped['Incomplete prior-week team history or first two weeks']+=1;continue
                vector,margin,total=features(data,game,result)
                rows.append(dict(id=target['id'],season=season,week=week,kickoff=target['kickoff'],feature_cutoff=cutoff.isoformat(),
                    away=target['away'],home=target['home'],features=vector,baseline_margin=margin,baseline_total=total,
                    actual_margin=target['away_score']-target['home_score'],actual_total=target['away_score']+target['home_score'],
                    line_away=target['line_away'],market_total=target['market_total'],
                    origin='Retrospective prior-week reconstruction; historical closing lines used only for evaluation.'))
            except (ValueError,TypeError,KeyError) as exc:skipped[str(exc)]+=1
        if progress:progress((i+1)/max(1,len(groups)),f'Built NFL {season}, week {week}')
    return rows,dict(skipped)
