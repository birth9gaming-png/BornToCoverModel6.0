"""Free live availability checks and additive score scenarios; no core-model edits.

Provider listings are partial observations, never proof that a team is healthy.
Default player values/status probabilities are editable priors, not trained effects.
"""
from __future__ import annotations
import copy
import json
import math
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from data_pipeline import normal_name, request_bytes, NFL_NAMES
from matchup_history import timestamp
from nfl_attd import name_key
from nfl_stats import code

ESPN = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries'
COVERS = {'NFL':'https://www.covers.com/sport/football/nfl/injuries',
          'CFB':'https://www.covers.com/sport/football/ncaaf/injuries'}
DEFAULTS = dict(enabled=True, NFL=True, CFB=True, score_adjustments=True,
                refresh_minutes=60, max_report_days=14, questionable_absence=.25,
                doubtful_absence=.75, nfl_cap=4., cfb_cap=1.5)
VALUES = {'QB':3.,'RB':.45,'FB':.10,'WR':.65,'TE':.40,'OT':.30,'OG':.20,'G':.20,
          'C':.25,'OL':.25,'DL':.20,'DT':.20,'DE':.25,'EDGE':.25,'LB':.15,
          'ILB':.15,'OLB':.15,'CB':.25,'DB':.20,'S':.15,'FS':.15,'SS':.15,'K':.15,'P':.10}
DEFENSE = {'DL','DT','DE','EDGE','LB','ILB','OLB','CB','DB','S','FS','SS'}


def settings(data):
    result={**DEFAULTS,**data.get('auto_injury_settings',{})}
    for key in ('enabled','NFL','CFB','score_adjustments'):
        if not isinstance(result[key],bool):raise ValueError('Injury switches must be Boolean.')
    for key in ('refresh_minutes','max_report_days','questionable_absence','doubtful_absence','nfl_cap','cfb_cap'):
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid automatic injury setting: '+key)
    if not 5<=result['refresh_minutes']<=1440 or not 1<=result['max_report_days']<=30:
        raise ValueError('Use injury refresh 5–1440 minutes and report age 1–30 days.')
    if any(result[k]>1 for k in ('questionable_absence','doubtful_absence')):
        raise ValueError('Injury absence probabilities must be 0–1.')
    return result


def status(value):
    value=str(value).strip().lower()
    if value in ('out','inactive','injured reserve','ir','pup','suspended') or value.startswith(('out ','out-','out for')):return 'Out'
    if value.startswith('doubtful'):return 'Doubtful'
    if value.startswith('questionable') or value in ('day-to-day','day to day'):return 'Questionable'
    if value in ('active','available','healthy'):return 'Available'
    # A practice injury, limited practice, or a probable designation is not an absence.
    return 'Unknown'


def parse_espn(payload, observed_at):
    if not isinstance(payload,dict) or not isinstance(payload.get('injuries'),list):raise ValueError('NFL injury feed changed schema.')
    rows=[];teams=[]
    for group in payload['injuries']:
        team=str(group.get('displayName','')).strip();teams.append(team)
        for item in group.get('injuries',[]):
            athlete=item.get('athlete') or {};name=athlete.get('displayName','')
            if not name:continue
            raw=str(item.get('status',''));details=item.get('details') or {}
            rows.append(dict(league='NFL',team=team,player=name,position=(athlete.get('position') or {}).get('abbreviation',''),
                status=status(raw),provider_status=raw,report_updated_at=item.get('date'),date_precision='timestamp',
                observed_at=observed_at,source='ESPN public injury listing',source_url=ESPN,
                injury=details.get('type',''),long_term=raw.lower() in ('injured reserve','ir','pup','suspended')))
    if not teams:raise ValueError('NFL injury feed contained no team groups.')
    return rows,teams


def parse_covers(html, league, observed_at):
    soup=BeautifulSoup(html,'html.parser');blocks=soup.select('.covers-CoversSeasonInjuries-blockContainer')
    if not blocks:raise ValueError('Public injury listing changed layout or returned no team sections.')
    now=timestamp(observed_at).astimezone(ZoneInfo('America/New_York'));rows=[];teams=[]
    for block in blocks:
        heading=block.select_one('.covers-CoversMatchups-teamName a')
        if heading is None:continue
        short=' '.join(str(s).strip() for s in heading.contents if isinstance(s,str)).strip()
        mascot=heading.select_one('span');full=short+(' '+mascot.get_text(' ',strip=True) if mascot else '')
        team=full if league=='NFL' else short;teams.append(team)
        for tr in block.select('tbody tr'):
            cells=tr.find_all('td',recursive=False)
            if len(cells)<3 or cells[0].get('colspan'):continue
            name=' '.join(cells[0].get_text(' ',strip=True).split());position=cells[1].get_text(' ',strip=True)
            bold=cells[2].find('b');description=bold.get_text(' ',strip=True) if bold else cells[2].get_text(' ',strip=True)
            date_match=re.search(r'\b([A-Z][a-z]{2})\s+(\d{1,2})\b',cells[2].get_text(' ',strip=True))
            reported=None
            if date_match:
                candidate=datetime.strptime(date_match.group(0)+' '+str(now.year),'%b %d %Y').date()
                if candidate>now.date()+timedelta(days=2):candidate=candidate.replace(year=now.year-1)
                reported=candidate.isoformat()
            detail=tr.find_next_sibling('tr');note=detail.get_text(' ',strip=True) if detail and 'collapse' in detail.get('class',[]) else ''
            rows.append(dict(league=league,team=team,player=name,position=position,status=status(description.split(' - ')[0]),
                provider_status=description,report_updated_at=reported,date_precision='day',observed_at=observed_at,
                source='Covers public injury listing',source_url=COVERS[league],injury=description,
                long_term=bool(re.search(r'season|indefinite|injured reserve',description+' '+note,re.I))))
    if not teams:raise ValueError('Injury listing contained no recognizable team names.')
    return rows,teams


def _team_match(name,data,league):
    lookup={}
    for tid,team in data.get('teams',{}).items():
        if team.get('league')!=league:continue
        names=[tid,team.get('name',tid)]
        if league=='NFL':names.append(NFL_NAMES.get(code(tid),''))
        for item in names:
            if item:lookup.setdefault(normal_name(item),set()).add(tid)
    matches=lookup.get(normal_name(name),set())
    return next(iter(matches)) if len(matches)==1 else None


def _recent(row,now,days):
    try:
        if row.get('date_precision')=='day':
            day=datetime.fromisoformat(row['report_updated_at']).date();current=now.astimezone(ZoneInfo('America/New_York')).date()
            return 0<=(current-day).days<=days
        at=timestamp(row.get('report_updated_at'))
        return timedelta(0)<=now-at<=timedelta(days=days)
    except (ValueError,TypeError):return False


def refresh(data,fetch=request_bytes,now=None,force=False):
    """Only check upcoming slates. Keep failures visible; never certify full coverage."""
    now=now or datetime.now(timezone.utc);config=settings(data)
    if not config['enabled']:return False
    any_change=False
    for league in ('NFL','CFB'):
        if not config[league]:continue
        games=[g for g in data.get('games',[]) if g.get('league')==league]
        upcoming=[]
        for game in games:
            try:
                if now<timestamp(game.get('kickoff'))<=now+timedelta(days=8):upcoming.append(game)
            except ValueError:pass
        if not upcoming:continue
        old=data.get('auto_injury_checks',{}).get(league,{})
        try:age=(now-timestamp(old.get('checked_at'))).total_seconds()
        except ValueError:age=float('inf')
        if not force and 0<=age<config['refresh_minutes']*60:continue
        observed=now.isoformat();notes=[];rows=[];team_names=[];source=None
        try:
            if league=='NFL':
                try:
                    rows,team_names=parse_espn(json.loads(fetch(ESPN,ttl=300)),observed)
                    if not any(_recent(r,now,config['max_report_days']) for r in rows):raise ValueError('NFL listing has no recent player reports.')
                    source='ESPN'
                except (RuntimeError,ValueError,TypeError,KeyError):
                    notes.append('Primary NFL listing unavailable or stale; used the free fallback.')
            if not source:
                rows,team_names=parse_covers(fetch(COVERS[league],ttl=300).decode('utf-8'),league,observed);source='Covers'
            mapped=[];stale=0;unmatched=0
            for row in rows:
                tid=_team_match(row['team'],data,league)
                if not tid:unmatched+=1;continue
                if not _recent(row,now,config['max_report_days']):stale+=1;continue
                row['team_id']=tid;mapped.append(row)
            covered=sorted({tid for name in team_names if (tid:=_team_match(name,data,league))})
            data['auto_injury_reports']=[r for r in data.get('auto_injury_reports',[]) if r.get('league')!=league]+mapped
            check=dict(checked_at=observed,state='Partial listing' if mapped else 'No recent matched reports',source=source,
                matched_players=len(mapped),listed_teams=covered,stale_rows_withheld=stale,notes=notes)
        except (RuntimeError,ValueError,TypeError,KeyError,OSError) as exc:
            check=dict(checked_at=observed,state='Unavailable',source=None,matched_players=0,listed_teams=[],notes=['Free injury source could not be read. Last reports will expire; no healthy-team assumption.'])
        data.setdefault('auto_injury_checks',{})[league]=check;any_change=True
    return any_change


def eligible_reports(data,game,now=None):
    now=now or datetime.now(timezone.utc);config=settings(data)
    if not config['enabled'] or not config.get(game.get('league'),False):return []
    try:kickoff=timestamp(game.get('kickoff'))
    except ValueError:return []
    # This layer is live only. Never apply a report observed after kickoff.
    if kickoff<=now:return []
    latest={}
    for row in data.get('auto_injury_reports',[]):
        if row.get('league')!=game['league'] or row.get('team_id') not in (game['away'],game['home']):continue
        try:at=timestamp(row.get('observed_at'))
        except ValueError:continue
        if not at<=now<at+timedelta(hours=24) or at>=kickoff or not _recent(row,now,config['max_report_days']):continue
        key=(row['team_id'],name_key(row['player']))
        order=(str(row.get('report_updated_at','')),at)
        if key not in latest or order>(str(latest[key].get('report_updated_at','')),timestamp(latest[key]['observed_at'])):latest[key]=row
    return list(latest.values())


def same_player(a,b):
    a,b=name_key(a).split(),name_key(b).split()
    return bool(a and b and (a==b or (a[-1]==b[-1] and a[0][0]==b[0][0] and (len(a[0])==1 or len(b[0])==1))))


def score_effects(data,game,now=None):
    config=settings(data);report=dict(away=0.,home=0.,rows=[],notes=[])
    rows=eligible_reports(data,game,now)
    if not config['score_adjustments']:report['notes'].append('Automatic score adjustments are off; availability checks remain active.');return report
    manual=[r for r in data.get('injury_reports',[]) if r.get('reviewed') and r.get('league')==game.get('league')]
    profiles=data.get('nfl_injury_players',[]) if game['league']=='NFL' else []
    for row in rows:
        side='away' if row['team_id']==game['away'] else 'home';entry=dict(row,own_score_change=0.,opponent_score_change=0.)
        recent_manual=[]
        for r in manual:
            try:
                at=timestamp(r.get('report_as_of'));kickoff=timestamp(game['kickoff'])
                if r.get('team_id')==row['team_id'] and same_player(r['player'],row['player']) and at<=min(now or datetime.now(timezone.utc),kickoff) and kickoff-at<=timedelta(hours=96):recent_manual.append(r)
            except ValueError:pass
        if recent_manual:entry['reason']='Reviewed manual row takes precedence';report['rows'].append(entry);continue
        probability={'Out':1.,'Doubtful':config['doubtful_absence'],'Questionable':config['questionable_absence'],'Available':0.}.get(row['status'])
        if probability is None:entry['reason']='Practice-only or unknown status; no absence inferred';report['rows'].append(entry);continue
        position=row.get('position','').upper()
        matches=[p for p in profiles if code(p.get('team'))==code(row['team_id']) and same_player(p.get('name',''),row['player']) and (not position or p.get('position','').upper()==position)]
        profile=matches[0] if len(matches)==1 else {}
        overrides=[p for p in data.get('injury_player_values',[]) if p.get('league')==game['league'] and p.get('team_id')==row['team_id'] and same_player(p.get('player',''),row['player'])]
        if game['league']=='NFL' and not profile.get('baseline_verified') and not (len(overrides)==1 and 'baseline_absence' in overrides[0]):
            entry['reason']='Recent snap/role baseline missing; availability shown, score term skipped';report['rows'].append(entry);continue
        if game['league']=='CFB' and row.get('long_term') and not overrides:
            entry['reason']='Long-term absence may already be in team statistics; score term skipped';report['rows'].append(entry);continue
        value=VALUES.get(position,0.)
        if len(overrides)==1:
            value=float(overrides[0].get('points',value));baseline=float(overrides[0].get('baseline_absence',profile.get('baseline_absence',0.)))
            scale=1.
        else:
            baseline=float(profile.get('baseline_absence',0.));scale=float(profile.get('role_share',.25))
        if not 0<=value<=10 or not 0<=baseline<=1 or not 0<=scale<=1:raise ValueError('Invalid injury player value, baseline or role share.')
        impact=value*scale*(probability-baseline)
        opposite='home' if side=='away' else 'away'
        if position in DEFENSE:report[opposite]+=impact;entry['opponent_score_change']=impact
        else:report[side]-=impact;entry['own_score_change']=-impact
        entry.update(absence_probability=probability,baseline_absence=baseline,role_share=scale,
                     reason='Incremental scenario relative to recent snaps' if profile else 'Conservative CFB role proxy; role/baseline unverified')
        report['rows'].append(entry)
    cap=config['nfl_cap'] if game['league']=='NFL' else config['cfb_cap']
    for side in ('away','home'):report[side]=max(-cap,min(cap,report[side]))
    report['notes'].append('Player values and uncertain-status probabilities are scenario priors. NFL baseline uses recent actual snaps; CFB uses a conservative role proxy. Missing players do not mean healthy coverage.')
    return report


def project(data,game,parameters=None):
    from nfl_layers import project as existing_project
    from model import evaluate, quotes_for, parameters_for_league
    result=existing_project(data,game,parameters)
    if 'away' not in result:return result
    effect=score_effects(data,game);result['automatic_injuries']=effect
    if not effect['away'] and not effect['home']:return result
    old_margin=result['margin_away'];old_total=result['fair_total']
    for side in ('away','home'):
        result[side]['score_parts']['automatic_injuries']=effect[side]
        result[side]['score']+=effect[side]
    extra=result.get('nfl_layers',{})
    if game['league']=='NFL' and extra.get('context',{}).get('divisional_matchup'):
        from nfl_layers import settings as nfl_settings
        config=nfl_settings(data)
        if config['enabled'] and config['division_enabled']:
            # Keep the requested division shrink on the final injury-adjusted
            # margin, while applying the fixed division total drop only once.
            compressed=-config['division_margin_shrink']*(effect['away']-effect['home'])
            for side,sign in (('away',1),('home',-1)):
                delta=sign*compressed/2
                result[side]['score_parts']['nfl_division_layer']+=delta
                result[side]['score']+=delta
                extra['terms']['nfl_division_layer'][side]+=delta
            extra['margin_adjustment']+=compressed
    if min(result[s]['score'] for s in ('away','home'))<=0:raise ValueError('Injury assumptions produce a nonpositive score.')
    result['margin_away']=result['away']['score']-result['home']['score'];result['fair_total']=result['away']['score']+result['home']['score']
    result['fair_line_away']=-result['margin_away'];result['raw_margin_away']+=result['margin_away']-old_margin;result['raw_total']+=result['fair_total']-old_total
    result['spread_edge_away']=result['margin_away']+game['line_away'] if game.get('line_away') is not None else None
    result['total_edge_over']=result['fair_total']-game['total'] if game.get('total') is not None else None
    old={(q['kind'],q['selection'],q.get('book'),q['line']):q.get('ats_priority_bonus',0.) for q in result['markets']}
    par=parameters_for_league(parameters or data['parameters'],game['league']);result['markets']=[];result['quote_errors']=[]
    for quote in quotes_for(game):
        try:
            q=evaluate(quote,result,par[game['league']]);bonus=old.get((q['kind'],q['selection'],q.get('book'),q['line']),0.)
            q.update(ats_priority_bonus=bonus,selection_score=q['conditional_ev']+bonus);result['markets'].append(q)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:result['quote_errors'].append(str(exc))
    result['best_market']=max(result['markets'],key=lambda q:q.get('selection_score',q['conditional_ev']),default=None)
    return result


def run(data,parameters=None):
    results=[]
    for game in data['games']:
        try:results.append(project(data,game,parameters))
        except (ValueError,ArithmeticError,TypeError,KeyError) as exc:results.append(dict(game=copy.deepcopy(game),league=game.get('league'),status=str(exc),markets=[],best_market=None))
    return results
