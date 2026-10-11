"""Compact, additive NFL game statistics shared by live and historical loaders."""
from __future__ import annotations
from collections import defaultdict
import math
import pandas as pd

ALIASES = {'OAK':'LV','SD':'LAC','STL':'LA','LAR':'LA','JAC':'JAX','WSH':'WAS'}
PBP_COLUMNS = {'game_id','season_type','week','posteam','defteam','play_type',
    'yards_gained','epa','pass_attempt','interception','no_play','qb_kneel','qb_spike',
    'qb_dropback','sack','cpoe','fumbled_1_team','fumbled_2_team',
    'fumble_recovery_1_team','fumble_recovery_2_team',
    'play_id','yardline_100','rusher_player_id','receiver_player_id','td_player_id',
    'rush_touchdown','pass_touchdown','touchdown','complete_pass','was_pressure','n_blitzers','is_blitz'}
REQUIRED = {'game_id','season_type','week','posteam','defteam','play_type',
    'yards_gained','epa','pass_attempt','interception','fumbled_1_team','fumble_recovery_1_team'}
NFL_METRICS = ('epa','depa','pass_epa','dpass_epa','rush_epa','drush_epa',
               'cpoe','sack_rate','dsack_rate')


def code(value):
    value = str(value).upper().removeprefix('NFL:')
    return ALIASES.get(value,value)


def game_summaries(pbp):
    """Keep sums and denominators, not target-game scores or season averages."""
    if REQUIRED-set(pbp.columns):
        raise ValueError('NFL play-by-play is missing: '+', '.join(sorted(REQUIRED-set(pbp.columns))))
    df=pbp[pbp['season_type'].isin(['REG','POST'])].copy()
    if df.empty:return []
    df['game_id']=df['game_id'].astype(str)
    for name in ('posteam','defteam','fumbled_1_team','fumbled_2_team','fumble_recovery_1_team','fumble_recovery_2_team'):
        if name in df:df[name]=df[name].map(lambda v:code(v) if pd.notna(v) else None)
    for name in ('week','yards_gained','epa','pass_attempt','interception','sack','cpoe','qb_dropback'):
        if name in df:df[name]=pd.to_numeric(df[name],errors='coerce')
    eligible=df['play_type'].isin(['run','pass']) & df['posteam'].notna() & df['defteam'].notna()
    for flag in ('no_play','qb_kneel','qb_spike'):
        if flag in df:eligible &= pd.to_numeric(df[flag],errors='coerce').fillna(0).eq(0)
    plays=df[eligible].copy()
    if plays.empty:return []
    passing=plays['pass_attempt'].eq(1)
    dropback=plays['qb_dropback'].eq(1) if 'qb_dropback' in plays else passing | (plays['sack'].eq(1) if 'sack' in plays else False)
    rushing=plays['play_type'].eq('run') & ~dropback
    cpoe=plays['cpoe'] if 'cpoe' in plays else pd.Series(float('nan'),index=plays.index)
    plays['plays']=1
    plays['yards']=plays['yards_gained'].fillna(0)
    plays['epa_sum']=plays['epa'].fillna(0)
    plays['successes']=plays['epa'].gt(0).astype(int)
    plays['pass_plays']=passing.astype(int)
    plays['explosive_passes']=(passing & plays['yards_gained'].ge(20)).astype(int)
    plays['dropbacks']=dropback.astype(int)
    plays['pass_epa_sum']=plays['epa'].where(dropback,0).fillna(0)
    plays['rush_plays']=rushing.astype(int)
    plays['rush_epa_sum']=plays['epa'].where(rushing,0).fillna(0)
    plays['sacks']=plays['sack'].eq(1).astype(int) if 'sack' in plays else 0
    plays['sack_observations']=plays['sack'].notna().where(dropback,False).astype(int) if 'sack' in plays else 0
    plays['cpoe_sum']=cpoe.where(dropback,0).fillna(0)
    plays['cpoe_count']=(dropback & cpoe.notna()).astype(int)
    plays['invalid_core_plays']=(plays['yards_gained'].isna() | plays['epa'].isna() | plays['pass_attempt'].isna()).astype(int)
    columns=['plays','yards','epa_sum','successes','pass_plays','explosive_passes','dropbacks',
             'pass_epa_sum','rush_plays','rush_epa_sum','sacks','sack_observations','cpoe_sum','cpoe_count','invalid_core_plays']
    grouped=plays.groupby(['game_id','posteam'],sort=True)[columns].sum()
    turnover=defaultdict(float)
    ints=df[df['interception'].eq(1)]
    for (gid,team),count in ints.groupby(['game_id','posteam']).size().items():turnover[(gid,team)]-=float(count)
    for (gid,team),count in ints.groupby(['game_id','defteam']).size().items():turnover[(gid,team)]+=float(count)
    for i in (1,2):
        a,b=f'fumbled_{i}_team',f'fumble_recovery_{i}_team'
        if a not in df or b not in df:continue
        lost=df[df[a].notna() & df[b].notna() & df[a].ne(df[b])]
        for (gid,team),count in lost.groupby(['game_id',a]).size().items():turnover[(gid,team)]-=float(count)
        for (gid,team),count in lost.groupby(['game_id',b]).size().items():turnover[(gid,team)]+=float(count)
    return [dict(id=gid,team=team,**{k:float(v) for k,v in row.items()},net_turnovers=turnover[(gid,team)])
            for (gid,team),row in grouped.iterrows()]


def team_features(summaries,games,names=None,through_week=None):
    """Games supplies the expected complete history; missing game stats withhold teams."""
    names=names or {}
    stats={(r['id'],r['team']):r for r in summaries}
    previous=defaultdict(list)
    for game in games:
        for side in ('away','home'):previous[game[side]].append(game)
    features={}
    for team,history in previous.items():
        offense=[];defense=[]
        for game in history:
            opponent=game['home'] if game['away']==team else game['away']
            own,allowed=stats.get((game['id'],team)),stats.get((game['id'],opponent))
            if not own or not allowed or own.get('invalid_core_plays',0)>0 or allowed.get('invalid_core_plays',0)>0:break
            offense.append(own);defense.append(allowed)
        if len(offense)!=len(history):continue
        off={k:sum(r.get(k,0) for r in offense) for k in offense[0] if k not in ('id','team')}
        de={k:sum(r.get(k,0) for r in defense) for k in defense[0] if k not in ('id','team')}
        if min(off['plays'],de['plays'],off['pass_plays'],de['pass_plays'])<=0:continue
        def ratio(r,n,d):return r[n]/r[d] if r.get(d,0)>0 else None
        features['NFL:'+team]=dict(league='NFL',name=names.get(team,team),
            y=off['yards']/off['plays'],dy=de['yards']/de['plays'],
            s=100*off['successes']/off['plays'],ds=100*de['successes']/de['plays'],
            x=100*off['explosive_passes']/off['pass_plays'],dx=100*de['explosive_passes']/de['pass_plays'],
            p=off['plays']/len(history),t=off['net_turnovers']/len(history),
            epa=ratio(off,'epa_sum','plays'),depa=ratio(de,'epa_sum','plays'),
            pass_epa=ratio(off,'pass_epa_sum','dropbacks'),dpass_epa=ratio(de,'pass_epa_sum','dropbacks'),
            rush_epa=ratio(off,'rush_epa_sum','rush_plays'),drush_epa=ratio(de,'rush_epa_sum','rush_plays'),
            cpoe=ratio(off,'cpoe_sum','cpoe_count'),
            sack_rate=100*ratio(off,'sacks','dropbacks') if off.get('sack_observations')==off.get('dropbacks') and off['dropbacks']>0 else None,
            dsack_rate=100*ratio(de,'sacks','dropbacks') if de.get('sack_observations')==de.get('dropbacks') and de['dropbacks']>0 else None,
            sample_games=len(history),sample_plays=off['plays'],feature_through_week=through_week,
            source='nflverse prior-week game summaries; EPA includes sacks and turnovers')
    return features


def live_features(pbp,through_week,names=None):
    df=pbp[(pbp['season_type']=='REG') & (pd.to_numeric(pbp['week'],errors='coerce')<=through_week)]
    summaries=game_summaries(df)
    teams=defaultdict(set)
    for row in summaries:teams[row['id']].add(row['team'])
    games=[dict(id=gid,away=sorted(codes)[0],home=sorted(codes)[1]) for gid,codes in sorted(teams.items()) if len(codes)==2]
    return team_features(summaries,games,names,through_week)


def rest_days(game):
    result={}
    for side in ('away','home'):
        value=game.get(side+'_rest')
        try:value=float(value)
        except (ValueError,TypeError):value=None
        result[side]=value if value is not None and math.isfinite(value) and 0<value<=30 else None
    return result
