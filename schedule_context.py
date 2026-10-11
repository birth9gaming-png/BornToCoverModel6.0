"""Dated schedule signals; their point effects are editable research assumptions."""
from __future__ import annotations
import math
from zoneinfo import ZoneInfo
from matchup_history import timestamp
from poll_history import ranking

SIGNALS = ('lookahead', 'big_win', 'road_loss_return_home')
CONTEXT_DEFAULTS = dict(lookahead_margin_points=0., big_win_margin_points=0.,
                        road_loss_home_margin_points=.5, lookahead_total_points=0.,
                        big_win_total_points=0., context_cap_points=2.,
                        context_window_days=10., major_opponent_points=4.)


def defaults(league):
    return {**CONTEXT_DEFAULTS, 'major_opponent_points':8. if league=='CFB' else 4.}


def validate_context(data):
    for league in ('NFL','CFB'):
        par={**defaults(league),**data['parameters'][league]}
        for key in CONTEXT_DEFAULTS:
            value=par[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Invalid schedule-context parameter: '+key)
        if par['context_cap_points']<=0 or not 1<=par['context_window_days']<=30 or par['major_opponent_points']<0:
            raise ValueError('Context cap must be positive; window is 1–30 days; strength threshold cannot be negative.')
    if not isinstance(data['parameters']['CFB'].get('context_ranked_only',False),bool):raise ValueError('Ranked-opponent context setting must be boolean.')
    if data['parameters']['CFB'].get('selection_style','standard') not in ('standard','fade_spots','avoid_spots'):raise ValueError('Unknown CFB selection style.')
    if not isinstance(data.get('schedule',[]),list):raise ValueError('Schedule must be a list.')
    for game in data['games']:
        overrides=game.get('context_overrides',{})
        if not isinstance(overrides,dict):raise ValueError('Context overrides must be an object.')
        for side,record in overrides.items():
            if side not in ('away','home') or not isinstance(record,dict):raise ValueError('Invalid context override side.')
            for key,value in record.items():
                if key not in SIGNALS or (value is not None and not isinstance(value,bool)):
                    raise ValueError('Context overrides must be boolean flags or null for automatic detection.')


def opponent_strength(data,team_id,game):
    """Neutral, reference-pace margin versus an average team, using frozen inputs."""
    team=data['teams'].get(team_id)
    if not team or team.get('league')!=game['league']:return None
    cutoff=game.get('feature_through_week');used=team.get('feature_through_week')
    if cutoff is not None and (used is None or used>cutoff):return None
    keys=('y','dy','s','ds','x','dx')
    if any(not isinstance(team.get(k),(int,float)) or not math.isfinite(team[k]) for k in keys):return None
    par=data['parameters'];lp=par[game['league']]
    core=lp.get('shrink',par['shrink'])*(lp.get('by',par['by'])*(team['y']-team['dy'])+lp.get('bs',par['bs'])*(team['s']-team['ds'])+lp['bx']*(team['x']-team['dx']))/2
    if game['league']=='CFB' and all(isinstance(team.get(k),(int,float)) and math.isfinite(team[k]) for k in ('ofei','dfei')):
        season=data.get('season');week=team.get('fei_through_week')
        if season is not None and team.get('fei_season')==season and week is not None and (cutoff is None or week<=cutoff):
            w=lp.get('fei_weight',.25)
            core=(1-w)*core+w*lp.get('fei_drives',12.)*(team['ofei']+team['dfei'])
    return core


def _major(data,row,opponent,game,par):
    strength=opponent_strength(data,opponent,game)
    if isinstance(row.get('major_game'),bool):return row['major_game'],strength,'Explicit game-importance tag'
    if game['league']=='CFB' and par.get('context_ranked_only'):
        through=game.get('feature_through_week')
        try:previous=timestamp(row['kickoff'])<timestamp(game['kickoff'])
        except (ValueError,KeyError):previous=False
        if previous:through=int(row['week'])-1 if row.get('week') is not None else -1
        rank,ranked,source=ranking(data,opponent,game,through)
        return ranked,strength,(f'Ranked #{rank}; ' if rank else '')+source
    if strength is None:return None,None,'Opponent strength unavailable at the feature cutoff'
    return strength>=par['major_opponent_points'],strength,'Frozen opponent-strength proxy'


def _days(later,earlier):
    zone=ZoneInfo('America/New_York')
    return (later.astimezone(zone).date()-earlier.astimezone(zone).date()).days


def _neighbors(data,game,team_id):
    try:at=timestamp(game.get('kickoff'))
    except ValueError:return None,None,None
    rows={}
    for row in data.get('schedule',[]):
        if not isinstance(row,dict) or row.get('league')!=game['league'] or team_id not in (row.get('away'),row.get('home')):continue
        if row.get('season')!=data.get('season'):continue
        if row.get('id') is not None and game.get('id') is not None and str(row['id'])==str(game['id']):continue
        try:played=timestamp(row.get('kickoff'))
        except ValueError:continue
        identity=str(row['id']) if row.get('id') is not None else (row.get('away'),row.get('home'),played.isoformat())
        rows.setdefault(identity,(played,row))
    before=[v for v in rows.values() if v[0]<at];after=[v for v in rows.values() if v[0]>at]
    return at,max(before,key=lambda v:v[0]) if before else None,min(after,key=lambda v:v[0]) if after else None


def signals(data,game,side,par):
    own=game[side];opponent=game['home' if side=='away' else 'away']
    at,previous,next_game=_neighbors(data,game,own)
    result=dict(team=own,lookahead=None,big_win=None,road_loss_return_home=False if side=='away' or game.get('neutral') else None,
                previous=None,next=None,reasons={},sources={key:'Automatic schedule detection' for key in SIGNALS})
    if at is None:
        result['reasons']={key:'Current kickoff missing; automatic context unavailable' for key in SIGNALS}
    else:
        current_major,_,_=_major(data,game,opponent,game,par)
        if next_game:
            played,row=next_game;opp=row['home'] if row['away']==own else row['away']
            gap=_days(played,at);major,strength,source=_major(data,row,opp,game,par)
            # Future results are deliberately never copied or read here.
            result['next']=dict(id=row.get('id'),opponent=opp,kickoff=row['kickoff'],days_until=gap,major=major,opponent_strength=strength,importance_source=source)
            if not 1<=gap<=par['context_window_days'] or major is False or current_major is True:result['lookahead']=False
            elif major is True and current_major is False:result['lookahead']=True
            result['reasons']['lookahead']='Next major matchup within the window, while the current opponent is below the major threshold' if result['lookahead'] else 'Next/current importance unknown, current game is major, or next game does not qualify'
        else:result['reasons']['lookahead']='No subsequent scheduled game available'
        if previous:
            played,row=previous;opp=row['home'] if row['away']==own else row['away'];gap=_days(at,played)
            valid=row.get('completed') is True
            cutoff=game.get('feature_through_week')
            if cutoff is not None and (row.get('week') is None or row['week']>cutoff):valid=False
            a,h=row.get('away_score'),row.get('home_score')
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (a,h)):valid=False
            margin=(a-h if row['away']==own else h-a) if valid else None
            major,strength,source=_major(data,row,opp,game,par)
            result['previous']=dict(id=row.get('id'),opponent=opp,kickoff=row['kickoff'],days_since=gap,venue='Neutral' if row.get('neutral') else 'Away' if row['away']==own else 'Home',margin=margin,won=margin>0 if margin is not None else None,major=major,opponent_strength=strength,importance_source=source,result_verified=valid)
            if valid:
                recent=1<=gap<=par['context_window_days']
                result['big_win']=False if not recent or margin<=0 or major is False else True if major is True else None
                result['road_loss_return_home']=bool(recent and margin<0 and row['away']==own and not row.get('neutral') and side=='home' and not game.get('neutral'))
            result['reasons']['big_win']='Previous completed win against a major opponent within the window' if result['big_win'] else 'Latest result is missing, not a qualifying win, or opponent importance is unknown'
            result['reasons']['road_loss_return_home']='Previous completed road loss followed by a home game within the window' if result['road_loss_return_home'] else 'Requires a recent completed road loss and a current home venue'
        else:
            result['reasons']['big_win']='No previous game available'
            result['reasons']['road_loss_return_home']='No previous game available'
    for key,value in game.get('context_overrides',{}).get(side,{}).items():
        if value is None:continue
        if value and key=='road_loss_return_home' and (side!='home' or game.get('neutral')):
            raise ValueError('Road-loss return-home signal requires the current home team at a non-neutral venue.')
        if value and key=='big_win' and result['previous'] and result['previous']['result_verified'] and result['previous']['margin']<=0:
            raise ValueError('The verified previous result was not a win; big-win override conflicts with it.')
        if value and key=='road_loss_return_home' and result['previous'] and result['previous']['result_verified'] and (result['previous']['margin']>=0 or result['previous']['venue']!='Away'):
            raise ValueError('The verified previous result was not a road loss.')
        result[key]=value;result['sources'][key]='Manual scenario flag'
        result['reasons'][key]='User override; verify the matchup context before interpreting it'
    return result


def adjustments(data,game):
    par={**defaults(game['league']),**data['parameters'][game['league']]}
    sides={side:signals(data,game,side,par) for side in ('away','home')}
    flag=lambda side,key:1. if sides[side][key] is True else 0.
    margin_terms={key:par[weight]*(flag('away',key)-flag('home',key)) for key,weight in [('lookahead','lookahead_margin_points'),('big_win','big_win_margin_points'),('road_loss_return_home','road_loss_home_margin_points')]}
    total_terms={key:par[weight]*(flag('away',key)+flag('home',key)) for key,weight in [('lookahead','lookahead_total_points'),('big_win','big_win_total_points')]}
    clip=lambda value:max(-par['context_cap_points'],min(par['context_cap_points'],value))
    return dict(**sides,margin_terms=margin_terms,total_terms=total_terms,margin_adjustment=clip(sum(margin_terms.values())),total_adjustment=clip(sum(total_terms.values())),parameters=par)
