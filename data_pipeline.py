"""Explicit-refresh NFL/CFB data adapters. No embedded API keys or guessed inputs."""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import time
import tempfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import pandas as pd
from model import METRICS, CFB_METRICS, ensure_cfb_defaults, validate
from cfb_metrics import attach_fei, attach_portal
from matchup_history import timestamp
from ats_history import attach_records
from nfl_stats import PBP_COLUMNS, live_features as nfl_live_features, code as nfl_code
from nfl_features_ext import engineer_efficiency
from nfl_feeds import player_bundle,add_pressure_charting,fetch_attd_quotes,fetch_wind_forecasts
from nfl_layers import settings as nfl_layer_settings

ROOT = Path(__file__).resolve().parent
CFBD = 'https://api.collegefootballdata.com'
NFL_SCHEDULE = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
NFL_PBP = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz'
NFL_NAMES = dict(ARI='Arizona Cardinals', ATL='Atlanta Falcons', BAL='Baltimore Ravens', BUF='Buffalo Bills', CAR='Carolina Panthers', CHI='Chicago Bears', CIN='Cincinnati Bengals', CLE='Cleveland Browns', DAL='Dallas Cowboys', DEN='Denver Broncos', DET='Detroit Lions', GB='Green Bay Packers', HOU='Houston Texans', IND='Indianapolis Colts', JAX='Jacksonville Jaguars', KC='Kansas City Chiefs', LA='Los Angeles Rams', LAC='Los Angeles Chargers', LV='Las Vegas Raiders', MIA='Miami Dolphins', MIN='Minnesota Vikings', NE='New England Patriots', NO='New Orleans Saints', NYG='New York Giants', NYJ='New York Jets', PHI='Philadelphia Eagles', PIT='Pittsburgh Steelers', SEA='Seattle Seahawks', SF='San Francisco 49ers', TB='Tampa Bay Buccaneers', TEN='Tennessee Titans', WAS='Washington Commanders')

_CACHE_STORE=None


def set_cache_store(store):
    global _CACHE_STORE
    _CACHE_STORE=store


def effective_ttl(url,params,ttl):
    # Published past seasons are reused for 90 days; live requests retain their TTL.
    year=(params or {}).get('year')
    match=re.search(r'(?:play_by_play_|roster_|snap_counts_)(\d{4})',url)
    if year is None and match:year=match.group(1)
    try:
        if int(year)<date.today().year and ttl>0:return max(ttl,90*86400)
    except (TypeError,ValueError):pass
    return ttl


def normal_name(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def field(row, camel, snake=None, default=None):
    return row.get(camel, row.get(snake or camel, default))


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def request_bytes(url, params=None, headers=None, ttl=1800):
    from background_work import report_progress,request_label
    report_progress(request_label(url,params))
    ttl=effective_ttl(url,params,ttl)
    query = '?' + urlencode(params) if params else ''
    target = url + query
    fingerprint = hashlib.sha256((target + (headers or {}).get('Authorization', '')).encode()).hexdigest()
    cache = ROOT / '.cache' / (fingerprint + '.bin')
    if cache.exists() and time.time() - cache.stat().st_mtime < ttl:
        return cache.read_bytes()
    if _CACHE_STORE is not None:
        content=_CACHE_STORE.cache_get(fingerprint,ttl)
        if content is not None:
            # Do not reset the age of an older remote download in a new local file.
            return content
    req = Request(target, headers={'User-Agent': 'FootballModelLab/1.0 academic research', **(headers or {})})
    try:
        with urlopen(req, timeout=45) as response:
            content = response.read()
    except HTTPError as exc:
        raise RuntimeError(f'{urlparse(url).hostname}: HTTP {exc.code}. Check the API key, endpoint access, and quota.') from None
    except (URLError, TimeoutError):
        raise RuntimeError(f'{urlparse(url).hostname}: network request failed.') from None
    report_progress(request_label(url,params))
    cache.parent.mkdir(exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=cache.parent)
    with os.fdopen(fd,'wb') as target_file:target_file.write(content)
    os.replace(temp,cache)
    if _CACHE_STORE is not None:_CACHE_STORE.cache_put(fingerprint,content)
    return content


def request_json(url, params=None, headers=None, ttl=1800):
    try:
        value = json.loads(request_bytes(url, params, headers, ttl))
    except json.JSONDecodeError:
        raise RuntimeError(f'{urlparse(url).hostname}: response was not valid JSON.') from None
    if not isinstance(value, list):
        raise RuntimeError(f'{urlparse(url).hostname}: expected a list response. Check API access and schema.')
    return value


def empty_team(league, name):
    return dict(league=league, name=name, **{k: None for k in METRICS})


def game_defaults(league, away, home, game_id, kickoff):
    return dict(league=league, away=away, home=home, id=str(game_id), kickoff=kickoff,
                quotes=[], line_away=None, total=None, consensus_line_away=None, consensus_total=None,
                neutral=False, rest_margin_home=0, injury_away=0, injury_home=0, weather_total=0,
                modifiers_verified=False, notes='Injury, weather, travel and extra-rest modifiers are unverified; defaults are zero.')


def nfl_features(pbp, through_week):
    """Eligible scrimmage plays; completed prior weeks only, not the forecast week."""
    result=nfl_live_features(pbp,through_week,NFL_NAMES)
    for team,extra in engineer_efficiency(pbp,through_week).items():
        if team in result:result[team].update(extra)
    return result


def load_nfl(season, start, end, fetch=request_bytes,layer_options=None):
    schedule = pd.read_csv(io.BytesIO(fetch(NFL_SCHEDULE)))
    days = pd.to_datetime(schedule['gameday']).dt.date
    chosen = schedule[(schedule['season'] == season) & (schedule['game_type'] == 'REG') & (days >= start) & (days <= end)]
    # The screen intentionally includes already completed games in the selected window;
    # their predictions still use only prior weeks. Current odds may be unavailable.
    if chosen.empty: return {}, [], {}, []
    cutoff = int(chosen['week'].min()) - 1
    columns = PBP_COLUMNS
    warnings = [];bundle={};pressure_notes=[]
    try:
        pbp = pd.read_csv(io.BytesIO(fetch(NFL_PBP.format(season=season), ttl=3600)), compression='gzip', usecols=lambda c:c in columns, low_memory=False)
        required = {'game_id','season_type','week','posteam','defteam','play_type','yards_gained','epa','pass_attempt','interception','fumbled_1_team','fumble_recovery_1_team'}
        if required - set(pbp.columns): raise RuntimeError('nflverse play-by-play is missing required columns.')
        if layer_options is not None:
            pbp,pressure_notes=add_pressure_charting(pbp,season,fetch,layer_options)
            bundle=player_bundle(pbp,schedule[schedule.season.eq(season)],season,cutoff,fetch,layer_options)
            bundle['nfl_layer_feed_notes'].extend(pressure_notes)
        teams = nfl_features(pbp, cutoff)
    except (RuntimeError, ValueError) as exc:
        teams = {}; warnings.append('NFL statistics unavailable: ' + str(exc))
    games, aliases = [], {}
    for _, row in chosen.iterrows():
        codes = [str(row['away_team']), str(row['home_team'])]
        for code in codes:
            tid='NFL:'+code; teams.setdefault(tid, empty_team('NFL', NFL_NAMES.get(code,code)))
            for alias in (code, NFL_NAMES.get(code, code)):
                aliases[normal_name(alias)] = tid
        g = game_defaults('NFL', 'NFL:'+codes[0], 'NFL:'+codes[1], row['game_id'], str(row['gameday'])+' '+str(row.get('gametime',''))+' America/New_York')
        g['season']=season;g['week']=int(row['week']);g['feature_through_week']=cutoff;g['game_type']='REG'
        g['roof']=str(row.get('roof','unknown')).lower()
        g['away_rest']=number(row.get('away_rest'));g['home_rest']=number(row.get('home_rest'))
        g['neutral']=str(row.get('location','')).lower()=='neutral'
        if pd.notna(row.get('away_score')) and pd.notna(row.get('home_score')):
            g['completed']=True;g['actual_away']=float(row['away_score']);g['actual_home']=float(row['home_score'])
        games.append(g)
    pairs={frozenset((g['away'],g['home'])) for g in games}
    rating_history=[]
    for _,row in schedule[(schedule['season']==season)&(schedule['game_type']=='REG')&(schedule['week']<=cutoff)&(days<start)&schedule['away_score'].notna()&schedule['home_score'].notna()].iterrows():
        clock=row.get('gametime');clock=clock if isinstance(clock,str) and clock else '00:00'
        rating_history.append(dict(league='NFL',season=season,id=str(row['game_id']),away='NFL:'+str(row['away_team']),home='NFL:'+str(row['home_team']),week=int(row['week']),kickoff=str(row['gameday'])+' '+clock+' America/New_York',away_score=float(row['away_score']),home_score=float(row['home_score']),neutral=str(row.get('location','')).lower()=='neutral',completed=True))
    history=[]
    for _,row in schedule[(schedule['season']>=season-5)&(days<start)&schedule['away_score'].notna()&schedule['home_score'].notna()].iterrows():
        a,h='NFL:'+str(row['away_team']),'NFL:'+str(row['home_team'])
        if frozenset((a,h)) in pairs:
            history.append(dict(league='NFL',away=a,home=h,id=str(row['game_id']),kickoff=str(row['gameday'])+' '+str(row.get('gametime','00:00'))+' America/New_York',away_score=float(row['away_score']),home_score=float(row['home_score']),neutral=str(row.get('location','')).lower()=='neutral',source= NFL_SCHEDULE))
    involved={g[side] for g in games for side in ('away','home')}
    context_schedule=[]
    for _,row in schedule[(schedule['season']==season)&(schedule['game_type']=='REG')].iterrows():
        a,h='NFL:'+str(row['away_team']),'NFL:'+str(row['home_team'])
        if not involved.intersection((a,h)):continue
        clock=row.get('gametime');clock=clock if isinstance(clock,str) and clock else '00:00'
        complete=pd.notna(row.get('away_score')) and pd.notna(row.get('home_score'))
        context_schedule.append(dict(league='NFL',season=season,id=str(row['game_id']),away=a,home=h,week=int(row['week']),kickoff=str(row['gameday'])+' '+clock+' America/New_York',neutral=str(row.get('location','')).lower()=='neutral',completed=bool(complete),away_score=float(row['away_score']) if complete else None,home_score=float(row['home_score']) if complete else None,source=NFL_SCHEDULE))
    for g in games:
        g['history']=history;g['context_schedule']=context_schedule;g['rating_history']=rating_history
    ats_rows=[];old_codes={'STL':'LA','SD':'LAC','OAK':'LV'}
    for _,row in schedule[(days<start)&schedule['away_score'].notna()&schedule['home_score'].notna()].iterrows():
        clock=row.get('gametime');clock=clock if isinstance(clock,str) and clock else '00:00'
        a,h=str(row['away_team']),str(row['home_team']);line=number(row.get('spread_line'))
        location=str(row.get('location','')).lower()
        ats_rows.append(dict(league='NFL',season=int(row['season']),week=int(row['week']),id=str(row['game_id']),away='NFL:'+old_codes.get(a,a),home='NFL:'+old_codes.get(h,h),kickoff=str(row['gameday'])+' '+clock+' America/New_York',away_score=float(row['away_score']),home_score=float(row['home_score']),home_handicap=-line if line is not None else None,market_total=number(row.get('total_line')),neutral=True if location=='neutral' else False if location=='home' else None,completed=True))
    attach_records(teams,ats_rows,str(start)+' 00:00 America/New_York','nflverse closing spread; historic franchise codes mapped to current teams')
    if games:
        games[0]['trend_history']=ats_rows
        if bundle:games[0]['nfl_player_bundle']=bundle
    return teams, games, aliases, warnings


def box_metrics(rows, allowed_ids):
    accum = defaultdict(lambda:dict(games=0, yards=0, plays=0, allowed_yards=0, allowed_plays=0, net_to=0))
    for game in rows:
        if str(game.get('id')) not in allowed_ids: continue
        entries = game.get('teams', [])
        if len(entries)!=2: continue
        parsed=[]
        for team in entries:
            stats={x.get('category'):x.get('stat') for x in team.get('stats', [])}
            attempts=number(stats.get('passingAttempts'))
            if attempts is None:
                text=stats.get('completionAttempts')
                if isinstance(text,str) and re.fullmatch(r'\d+\s*[-/]\s*\d+',text):
                    attempts=number(re.split(r'[-/]',text)[-1].strip())
            rushes, yards, turnovers = number(stats.get('rushingAttempts')),number(stats.get('totalYards')),number(stats.get('turnovers'))
            if any(x is None for x in (attempts,rushes,yards,turnovers)):
                parsed=[];break
            parsed.append((field(team,'school','team'), yards, attempts+rushes, turnovers))
        if len(parsed)!=2 or any(x[2]<=0 for x in parsed): continue
        for i,(name,yards,plays,giveaways) in enumerate(parsed):
            opponent=parsed[1-i];a=accum[name]
            a['games']+=1;a['yards']+=yards;a['plays']+=plays;a['allowed_yards']+=opponent[1];a['allowed_plays']+=opponent[2];a['net_to']+=opponent[3]-giveaways
    return accum


def load_cfb(season, start, end, api_key, fetch=request_json):
    if not api_key: raise RuntimeError('A CollegeFootballData API key is required for CFB data.')
    headers={'Authorization':'Bearer '+api_key}
    schedule=fetch(CFBD+'/games',dict(year=season,seasonType='regular',classification='fbs'),headers)
    chosen=[]
    for g in schedule:
        kickoff=field(g,'startDate','start_date')
        if not kickoff: continue
        day=datetime.fromisoformat(kickoff.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date()
        if start<=day<=end: chosen.append(g)
    if not chosen:return {},[],{},[]
    cutoff=int(min(g['week'] for g in chosen))-1
    prior=[g for g in schedule if int(g['week'])<=cutoff and field(g,'homePoints','home_points') is not None and field(g,'awayPoints','away_points') is not None]
    allowed_ids={str(g['id']) for g in prior}
    weeks=sorted({int(g['week']) for g in prior})
    warnings=[];advanced=[];boxes=[]
    try:
        if cutoff>=0:
            advanced=fetch(CFBD+'/stats/season/advanced',dict(year=season,startWeek=0,endWeek=cutoff,excludeGarbageTime='false',classification='fbs'),headers)
        for week in weeks:
            boxes.extend(fetch(CFBD+'/games/teams',dict(year=season,week=week,seasonType='regular',classification='fbs'),headers,ttl=86400))
    except RuntimeError as exc:
        warnings.append('CFB statistics incomplete: '+str(exc))
    raw=box_metrics(boxes,allowed_ids)
    expected_counts=defaultdict(int)
    for g in prior:
        for side in ('home','away'):expected_counts[field(g,side+'Team',side+'_team')]+=1
    teams={}
    for row in advanced:
        name=row['team'];box=raw.get(name);o,d=row.get('offense',{}),row.get('defense',{})
        values=[field(o,'successRate','success_rate'),field(d,'successRate','success_rate'),o.get('explosiveness'),d.get('explosiveness')]
        # Do not mix a partial box-score sample with a complete advanced-stat sample.
        if not box or box['games']!=expected_counts[name] or any(number(x) is None for x in values):continue
        teams['CFB:'+name]=dict(league='CFB',name=name,y=box['yards']/box['plays'],dy=box['allowed_yards']/box['allowed_plays'],s=100*float(values[0]),ds=100*float(values[1]),x=float(values[2]),dx=float(values[3]),p=box['plays']/box['games'],t=box['net_to']/box['games'],sample_games=box['games'],sample_plays=box['plays'],feature_through_week=cutoff,source='CFBD prior-week advanced stats and box scores')
    aliases={}
    try:
        for team in fetch(CFBD+'/teams/fbs',dict(year=season),headers,ttl=86400):
            school=team['school'];tid='CFB:'+school
            names=[school, school+' '+str(team.get('mascot','')),team.get('abbreviation')]+(field(team,'alternateNames','alternate_names',[]) or [])
            for alias in names:
                if alias:
                    key=normal_name(alias)
                    if key in aliases and aliases[key]!=tid:aliases[key]=None
                    else:aliases[key]=tid
    except RuntimeError as exc:warnings.append('CFB name aliases unavailable: '+str(exc))
    games=[]
    for row in chosen:
        a,h=field(row,'awayTeam','away_team'),field(row,'homeTeam','home_team')
        for name in (a,h):
            tid='CFB:'+name;teams.setdefault(tid,empty_team('CFB',name));aliases[normal_name(name)]=tid
        g=game_defaults('CFB','CFB:'+a,'CFB:'+h,row['id'],field(row,'startDate','start_date'))
        g['week']=row['week'];g['feature_through_week']=cutoff;g['neutral']=bool(field(row,'neutralSite','neutral_site',False))
        if field(row,'awayPoints','away_points') is not None and field(row,'homePoints','home_points') is not None:
            g['completed']=True;g['actual_away']=field(row,'awayPoints','away_points');g['actual_home']=field(row,'homePoints','home_points')
        games.append(g)
    for team in teams.values():
        for key in CFB_METRICS: team.setdefault(key,None)
    try:
        count,fei_week=attach_fei(teams,request_bytes(f'https://bcftoys.com/{season}-fei',ttl=3600).decode('utf-8'),season,cutoff)
        if fei_week < cutoff: warnings.append(f'FEI is through week {fei_week}; other statistics use cutoff week {cutoff}.')
    except (RuntimeError,ValueError,UnicodeError) as exc:
        warnings.append('FEI unavailable: '+str(exc))
    try:
        starts=[datetime.fromisoformat(field(g,'startDate','start_date').replace('Z','+00:00')) for g in schedule if field(g,'startDate','start_date')]
        preseason_cutoff=min(starts)
        portal=fetch(CFBD+'/player/portal',dict(year=season),headers,ttl=86400)
        attach_portal(teams,portal,season,preseason_cutoff)
    except (RuntimeError,ValueError,TypeError) as exc:
        warnings.append('CFB portal ratings unavailable: '+str(exc))
    # Optional roster coverage is recalculated for slate teams in the dashboard,
    # including after manual overrides, rather than saved as a static warning.
    pairs={frozenset((g['away'],g['home'])) for g in games}
    rating_history=[]
    for row in prior:
        kickoff=field(row,'startDate','start_date')
        if not kickoff or row.get('completed') is False or timestamp(kickoff).astimezone(ZoneInfo('America/New_York')).date()>=start:continue
        rating_history.append(dict(league='CFB',season=season,id=str(row['id']),away='CFB:'+str(field(row,'awayTeam','away_team')),home='CFB:'+str(field(row,'homeTeam','home_team')),week=int(row['week']),kickoff=kickoff,away_score=float(field(row,'awayPoints','away_points')),home_score=float(field(row,'homePoints','home_points')),neutral=bool(field(row,'neutralSite','neutral_site',False)),completed=True))
    history=[];missing_ats_years=set()
    histories=list(schedule)
    for old_year in range(season-5,season):
        try: histories.extend(fetch(CFBD+'/games',dict(year=old_year,seasonType='both',classification='fbs'),headers,ttl=86400))
        except RuntimeError as exc:
            missing_ats_years.add(old_year);warnings.append(f'CFB history {old_year} unavailable: '+str(exc))
    ats_games=list(histories)
    for old_year in range(season-10,season-5):
        try:ats_games.extend(fetch(CFBD+'/games',dict(year=old_year,seasonType='both',classification='fbs'),headers,ttl=86400))
        except RuntimeError:missing_ats_years.add(old_year)
    venues={str(row['id']):row for row in ats_games}
    ats_rows=[];historic_quotes={}
    for old_year in range(season-10,season+1):
        try:line_games=fetch(CFBD+'/lines',dict(year=old_year,seasonType='both'),headers,ttl=86400)
        except RuntimeError:
            missing_ats_years.add(old_year);continue
        for row in line_games:
            if old_year==season and (row.get('week') is None or int(row['week'])>cutoff):continue
            a,h=field(row,'awayTeam','away_team'),field(row,'homeTeam','home_team')
            if not a or not h:continue
            spread_values=[]
            for quote in row.get('lines',[]):
                spread=verified_cfb_home_handicap(quote,a,h,aliases)
                if spread is not None:spread_values.append(spread)
            historic_quotes[str(row['id'])]=dict(away=a,home=h,values=spread_values,totals=[number(q.get('overUnder')) for q in row.get('lines',[]) if number(q.get('overUnder')) is not None])
    # Start from the full schedule, including unlisted/no-line games, so the
    # coverage percentage cannot improve just by omitting unavailable spreads.
    for row in venues.values():
        kickoff=field(row,'startDate','start_date')
        if not kickoff:continue
        if timestamp(kickoff).year==season and (row.get('week') is None or int(row['week'])>cutoff):continue
        a,h=field(row,'awayTeam','away_team'),field(row,'homeTeam','home_team')
        if not a or not h:continue
        quotes=historic_quotes.get(str(row['id']),{});values=quotes.get('values',[])
        if normal_name(quotes.get('away'))!=normal_name(a) or normal_name(quotes.get('home'))!=normal_name(h):values=[]
        neutral=field(row,'neutralSite','neutral_site')
        totals=quotes.get('totals',[])
        ats_rows.append(dict(league='CFB',season=int(row.get('season',timestamp(kickoff).year)),week=row.get('week'),id=str(row['id']),away='CFB:'+a,home='CFB:'+h,kickoff=kickoff,away_score=field(row,'awayPoints','away_points'),home_score=field(row,'homePoints','home_points'),home_handicap=median(values) if values else None,market_total=median(totals) if totals else None,neutral=neutral if isinstance(neutral,bool) else None,completed=row.get('completed',True)))
    attach_records(teams,ats_rows,str(start)+' 00:00 America/New_York','CFBD historical provider median; spread sign verified from formatted team handicap; neutral games excluded',sorted(missing_ats_years))
    if missing_ats_years:warnings.append('CFB ten-year ATS history is incomplete for seasons '+', '.join(map(str,sorted(missing_ats_years)))+'. Available records are displayed; the ATS adjustment is skipped until coverage is complete.')
    for row in histories:
        a,h='CFB:'+str(field(row,'awayTeam','away_team')),'CFB:'+str(field(row,'homeTeam','home_team'))
        if frozenset((a,h)) not in pairs: continue
        ka=field(row,'startDate','start_date');sa=field(row,'awayPoints','away_points');sh=field(row,'homePoints','home_points')
        if ka is None or sa is None or sh is None: continue
        try:
            if timestamp(ka).astimezone(ZoneInfo('America/New_York')).date()>=start: continue
        except ValueError: continue
        history.append(dict(league='CFB',away=a,home=h,id=str(row['id']),kickoff=ka,away_score=float(sa),home_score=float(sh),neutral=bool(field(row,'neutralSite','neutral_site',False)),source='CFBD /games'))
    involved={g[side] for g in games for side in ('away','home')}
    context_schedule=[]
    for row in schedule:
        a,h='CFB:'+str(field(row,'awayTeam','away_team')),'CFB:'+str(field(row,'homeTeam','home_team'))
        if not involved.intersection((a,h)):continue
        context_schedule.append(dict(league='CFB',season=season,id=str(row['id']),away=a,home=h,week=row['week'],kickoff=field(row,'startDate','start_date'),neutral=bool(field(row,'neutralSite','neutral_site',False)),completed=row.get('completed') is True,away_score=field(row,'awayPoints','away_points'),home_score=field(row,'homePoints','home_points'),source='CFBD /games'))
    for g in games:
        g['history']=history;g['context_schedule']=context_schedule;g['rating_history']=rating_history
    if games:
        games[0]['trend_history']=ats_rows
        try:
            from poll_history import snapshots
            games[0]['poll_history']=snapshots(fetch(CFBD+'/rankings',dict(year=season,seasonType='regular'),headers,ttl=3600))
        except (RuntimeError,ValueError,TypeError) as exc:warnings.append('CFB AP rankings unavailable: '+str(exc))
    return teams,games,aliases,warnings


def verified_cfb_home_handicap(quote, away, home, aliases):
    """Verify the handicap direction from the provider's named formatted line."""
    spread=number(quote.get('spread'))
    if spread==0:return 0.
    match=re.fullmatch(r'\s*(.*?)\s*([+-]\d+(?:\.\d+)?)\s*',str(field(quote,'formattedSpread','formatted_spread','')))
    if not match or spread is None:return None
    name,value=match.group(1),float(match.group(2))
    if abs(abs(value)-abs(spread))>1e-8:return None
    key=normal_name(name);tid=aliases.get(key)
    if key==normal_name(home) or (tid is not None and tid=='CFB:'+home):return value
    if key==normal_name(away) or (tid is not None and tid=='CFB:'+away):return -value
    return None


def attach_odds(games, events, aliases, start=None, end=None, diagnostics=None):
    """Join exact aliases AND New York game date; report only in-window failures."""
    local_zone=ZoneInfo('America/New_York')
    dated_games=[]
    for game in games:
        try: day=timestamp(game.get('kickoff')).astimezone(local_zone).date()
        except (ValueError,TypeError): continue
        dated_games.append((game,day))
    if start is None and dated_games: start=min(day for _,day in dated_games)
    if end is None and dated_games: end=max(day for _,day in dated_games)
    report=dict(received_events=len(events),outside_window=0,matched_events=0,matched_without_quotes=0,unmatched_events=[])
    unmatched=[]
    def miss(event, reason):
        name=str(event.get('away_team','?'))+' @ '+str(event.get('home_team','?'))
        unmatched.append(name)
        report['unmatched_events'].append(dict(event_id=event.get('id'),away_team=event.get('away_team'),home_team=event.get('home_team'),kickoff=event.get('commence_time'),reason=reason))
    for event in events:
        try: event_day=timestamp(event.get('commence_time')).astimezone(local_zone).date()
        except (ValueError,TypeError):
            miss(event,'Missing or invalid odds kickoff; date cannot be verified');continue
        # Filter before resolving names. The provider may list other weeks or
        # non-slate teams; those events are not failures in this selected window.
        if (start is not None and event_day<start) or (end is not None and event_day>end):
            report['outside_window']+=1;continue
        a,h=aliases.get(normal_name(event.get('away_team'))),aliases.get(normal_name(event.get('home_team')))
        if a is None or h is None:
            unknown=[str(event.get(side+'_team','?')) for side,tid in [('away',a),('home',h)] if tid is None]
            miss(event,'Unmapped or ambiguous team name: '+', '.join(unknown));continue
        candidates=[g for g,day in dated_games if g['away']==a and g['home']==h and day==event_day]
        if len(candidates)!=1:
            miss(event,'No scheduled matchup on this date' if not candidates else 'Multiple scheduled matchups on this date');continue
        g=candidates[0]
        if g.get('league')=='NFL':g['odds_event_id']=event.get('id')
        report['matched_events']+=1
        quote_count=len(g['quotes'])
        for book in event.get('bookmakers',[]):
            for market in book.get('markets',[]):
                kind={'spreads':'spread','totals':'total'}.get(market.get('key'))
                if not kind:continue
                for outcome in market.get('outcomes',[]):
                    label=outcome.get('name')
                    if kind=='spread':selection='away' if label==event['away_team'] else 'home' if label==event['home_team'] else None
                    else:selection={'Over':'over','Under':'under'}.get(label)
                    line,odds=number(outcome.get('point')),number(outcome.get('price'))
                    if selection and line is not None and odds is not None:
                        g['quotes'].append(dict(kind=kind,selection=selection,line=line,odds=odds,book=book.get('title') or book.get('key','Unnamed book'),updated_at=market.get('last_update',book.get('last_update'))))
        if len(g['quotes'])==quote_count: report['matched_without_quotes']+=1
        away_lines=[q['line'] for q in g['quotes'] if q['selection']=='away']
        over_lines=[q['line'] for q in g['quotes'] if q['selection']=='over']
        g['line_away']=median(away_lines) if away_lines else None;g['total']=median(over_lines) if over_lines else None
        g['consensus_line_away']=g['line_away'];g['consensus_total']=g['total'];g['book']='Median reference (not tradable)'
    if diagnostics is not None: diagnostics.update(report)
    return unmatched


def build_snapshot(season, start, end, leagues=('NFL','CFB'), cfbd_key='', odds_key='', regions='us', odds_alias_overrides=None,nfl_options=None):
    if end<start or (end-start).days>30:raise ValueError('Choose an ordered date window of at most 31 days.')
    baseline=json.loads((ROOT/'snapshot.json').read_text())
    out=dict(mode='connected',as_of=datetime.now(timezone.utc).isoformat(timespec='seconds'),snapshot_date=date.today().isoformat(),window=f'{start} through {end} America/New_York',parameters=copy.deepcopy(baseline['parameters']),teams={},games=[],warnings=[],sources={'NFL schedule':NFL_SCHEDULE,'NFL play-by-play':NFL_PBP.format(season=season),'CFB':'https://collegefootballdata.com/','Market quotes':'https://the-odds-api.com/liveapi/guides/v4/'},consensus_method='Median quoted away handicaps and over totals are comparison summaries only. EV is evaluated on each named book quote.')
    out['odds_join_diagnostics']={}
    nfl_options=nfl_layer_settings({'nfl_layer_settings':nfl_options or {}})
    out['nfl_layer_settings']=copy.deepcopy(nfl_options)
    out['odds_alias_overrides']=copy.deepcopy(odds_alias_overrides or {})
    local_zone=ZoneInfo('America/New_York')
    first=datetime.combine(start,datetime.min.time(),tzinfo=local_zone)
    last=datetime.combine(end,datetime.max.time().replace(microsecond=0),tzinfo=local_zone)
    odds_dates=dict(commenceTimeFrom=first.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),commenceTimeTo=last.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    for league in leagues:
        try:
            teams,games,aliases,warnings=load_nfl(season,start,end,layer_options=nfl_options) if league=='NFL' else load_cfb(season,start,end,cfbd_key)
        except (RuntimeError, ValueError, KeyError) as exc:
            out['warnings'].append(league+' data unavailable: '+str(exc));continue
        out['teams'].update(teams);out['games'].extend(games);out['warnings'].extend(warnings)
        if league=='NFL' and games:
            out.update(games[0].pop('nfl_player_bundle',{}))
            weather,notes=fetch_wind_forecasts(games,request_bytes,nfl_options,out['as_of'])
            out['nfl_weather_inputs']=weather;out.setdefault('nfl_layer_feed_notes',[]).extend(notes)
        for name,tid in out['odds_alias_overrides'].get(league,{}).items():
            if isinstance(name,str) and isinstance(tid,str) and tid in teams and teams[tid].get('league')==league:
                aliases[normal_name(name)]=tid
            else:out['warnings'].append(f'{league}: saved odds name mapping {name!r} does not match a loaded team. Review Data coverage and fixes.')
        if odds_key and games:
            sport='americanfootball_nfl' if league=='NFL' else 'americanfootball_ncaaf'
            try:
                events=request_json('https://api.the-odds-api.com/v4/sports/'+sport+'/odds',dict(apiKey=odds_key,regions=regions,markets='spreads,totals',oddsFormat='american',**odds_dates),ttl=120)
                diagnostic={};missed=attach_odds(games,events,aliases,start,end,diagnostic)
                if league=='NFL':
                    quotes,notes=fetch_attd_quotes(games,out.get('nfl_player_candidates',[]),odds_key,regions,request_bytes,nfl_options,out['as_of'])
                    out['nfl_attd_quotes']=quotes;out.setdefault('nfl_layer_feed_notes',[]).extend(notes)
                out['odds_join_diagnostics'][league]=diagnostic
                if missed:out['warnings'].append(f'{league}: {len(missed)} odds events inside your date window did not match the schedule uniquely. See Setup & Saved Data → Data coverage and fixes for the names and reasons.')
            except (RuntimeError,ValueError,KeyError) as exc:out['warnings'].append(league+' odds unavailable: '+str(exc))
        elif games:out['warnings'].append(league+': no Odds API key. Score projections are available where statistics are complete; market EV is unavailable.')
    if not out['games']:raise RuntimeError('No schedule games were returned. '+(' '.join(out['warnings']) or 'Try another date window.'))
    history=[];seen=set();context_schedule=[];schedule_seen=set();rating_history=[];rating_seen=set()
    out['trend_history']=[];out['poll_history']=[]
    for game in out['games']:
        out['trend_history'].extend(game.pop('trend_history',[]))
        out['poll_history'].extend(game.pop('poll_history',[]))
        for row in game.pop('history',[]):
            key=(row['league'],row['id'])
            if key not in seen: history.append(row);seen.add(key)
        for row in game.pop('context_schedule',[]):
            key=(row['league'],row['id'])
            if key not in schedule_seen:context_schedule.append(row);schedule_seen.add(key)
        for row in game.pop('rating_history',[]):
            key=(row['league'],row['id'])
            if key not in rating_seen:rating_history.append(row);rating_seen.add(key)
    out['history']=history
    out['schedule']=context_schedule
    out['rating_history']=rating_history
    out['season']=season
    ensure_cfb_defaults(out)
    out['sources']['FEI']='https://bcftoys.com/'+str(season)+'-fei'
    out['sources']['Transfer portal']='CFBD /player/portal; confirmed immediate preseason transfers; sum of native ratings in minus out'
    out['sources']['Returning starters']='User supplied; separate offensive/defensive counts, not inferred from returning production'
    out['sources']['Schedule context']='Current-season schedule/results from nflverse and CFBD; importance uses frozen opponent strength or manual flags'
    out['sources']['Opponent-adjusted result ratings']='Ridge least squares on current-season completed margins through the feature cutoff; schedules from nflverse and CFBD'
    out['sources']['Ten-year ATS']='NFL: nflverse closing spread. CFB: median verified provider handicap in CFBD historical /lines, joined to /games for venue. Rolling ten calendar years through the slate start; neutral games excluded.'
    validate(out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',type=int,required=True)
    parser.add_argument('--start',type=date.fromisoformat,required=True)
    parser.add_argument('--end',type=date.fromisoformat,required=True)
    parser.add_argument('--leagues',nargs='+',choices=['NFL','CFB'],default=['NFL','CFB'])
    parser.add_argument('--output',type=Path,default=Path('full_slate.json'))
    args=parser.parse_args()
    snapshot=build_snapshot(args.season,args.start,args.end,args.leagues,os.getenv('CFBD_API_KEY',''),os.getenv('ODDS_API_KEY',''))
    args.output.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
    print(f'Saved {len(snapshot["games"])} games to {args.output}.')
    for warning in snapshot['warnings']:print(warning)
