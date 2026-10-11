"""Additional NFL features; no changes to the core model or its fitted features.

Rate inputs use fractions (0.30 = 30%); DVOA inputs use percentage points
(15 = +15%). Defensive DVOA retains the provider convention: negative is good.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from audit_time import observed_time
from nfl_stats import code

DIVISIONS = {
    **dict.fromkeys(('BUF','MIA','NE','NYJ'),'AFC East'),
    **dict.fromkeys(('BAL','CIN','CLE','PIT'),'AFC North'),
    **dict.fromkeys(('HOU','IND','JAX','TEN'),'AFC South'),
    **dict.fromkeys(('DEN','KC','LAC','LV'),'AFC West'),
    **dict.fromkeys(('DAL','NYG','PHI','WAS'),'NFC East'),
    **dict.fromkeys(('CHI','DET','GB','MIN'),'NFC North'),
    **dict.fromkeys(('ATL','CAR','NO','TB'),'NFC South'),
    **dict.fromkeys(('ARI','LA','SF','SEA'),'NFC West'),
}


def utc(value):
    """Refuse to guess the timezone of an observation."""
    try:return pd.Timestamp(observed_time(value))
    except (ValueError,TypeError):return pd.NaT


def numeric(frame, key, default=np.nan):
    return pd.to_numeric(frame[key],errors='coerce') if key in frame else pd.Series(default,index=frame.index,dtype=float)


def eligible_plays(pbp, through_week):
    """Forecast-week and future plays never enter a feature."""
    df=pbp.copy()
    required={'posteam','defteam','play_type','week','epa','yards_gained'}
    if required-set(df):raise ValueError('PBP missing '+', '.join(sorted(required-set(df))))
    keep=numeric(df,'week').le(through_week) & df['play_type'].isin(['run','pass'])
    if 'season_type' in df:keep &= df['season_type'].isin(['REG','POST'])
    for flag in ('no_play','qb_kneel','qb_spike'):keep &= numeric(df,flag,0).fillna(0).eq(0)
    df=df[keep & df.posteam.notna() & df.defteam.notna()].copy()
    for key in ('posteam','defteam'):df[key]=df[key].map(code)
    df['dropback']=numeric(df,'qb_dropback',0).eq(1) | numeric(df,'pass_attempt',0).eq(1) | numeric(df,'sack',0).eq(1)
    df['designed_rush']=df.play_type.eq('run') & ~df.dropback
    return df


def engineer_efficiency(pbp, through_week, success_prior_plays=100, success_prior=.45):
    """Separate dropback/rush EPA, smooth success, and use actual pressure tags.

    `line_yards_proxy` is an unadjusted PBP run-blocking proxy, not official ALY.
    Official opponent-adjusted line yards can override it through the import.
    A sack/hit is never silently relabeled as a pressure or a non-blitz play.
    """
    df=eligible_plays(pbp,through_week)
    if df.empty:return {}
    epa=numeric(df,'epa');yards=numeric(df,'yards_gained')
    df['epa_value']=epa
    df['successful']=epa.gt(0).where(epa.notna())
    # Designed runs, not QB scrambles. Negative yards receive 120% attribution;
    # yards 5–10 receive 50%; yards after 10 receive no line attribution.
    df['line_yards']=np.where(yards.lt(0),1.2*yards,np.minimum(yards,4)+.5*np.clip(yards-4,0,6))
    pressure=numeric(df,'was_pressure')
    if 'was_pressure' in df:
        pressure=df.was_pressure.replace({'TRUE':1,'FALSE':0,'true':1,'false':0}).pipe(pd.to_numeric,errors='coerce')
    blitz=numeric(df,'n_blitzers')
    if 'is_blitz' in df:blitz=numeric(df,'is_blitz')
    df['non_blitz_observed']=df.dropback & blitz.eq(0) & pressure.isin([0,1])
    df['non_blitz_pressure']=pressure.where(df.non_blitz_observed)
    output={}
    for team,own in df.groupby('posteam'):
        opp=df[df.defteam.eq(team)]
        def mean(frame,column,mask=None):
            values=frame.loc[mask,column] if mask is not None else frame[column]
            return float(values.mean()) if values.notna().any() else None
        n=int(own.epa_value.notna().sum());success=float(own.successful.sum())
        nonblitz=opp[opp.non_blitz_observed]
        output['NFL:'+team]=dict(
            pass_epa=mean(own,'epa_value',own.dropback),rush_epa=mean(own,'epa_value',own.designed_rush),
            dpass_epa=mean(opp,'epa_value',opp.dropback),drush_epa=mean(opp,'epa_value',opp.designed_rush),
            success_rate=mean(own,'successful'),defensive_success_rate=mean(opp,'successful'),
            smoothed_success_rate=(success+success_prior_plays*success_prior)/(n+success_prior_plays),
            line_yards_proxy=mean(own,'line_yards',own.designed_rush),
            defensive_line_yards_proxy=mean(opp,'line_yards',opp.designed_rush),
            line_yards_source='PBP proxy; not opponent-adjusted ALY',
            non_blitz_pressure_rate=mean(nonblitz,'non_blitz_pressure'),
            non_blitz_dropbacks=int(len(nonblitz)),
            pass_share=float(own.dropback.mean()),sample_plays=n,
            feature_through_week=int(through_week))
    return output


def engineer_game_context(games, schedule=None):
    """Vectorized division identity, signed rest, and schedule-confirmed byes.

    rest_differential_home = home rest − away rest. A long rest interval alone
    is insufficient for a bye. A bye requires the complete REG team schedule
    and no scheduled game in the previous league week, or an explicit flag.
    Current division mapping is valid from the 2002 realignment onward.
    """
    df=pd.DataFrame(games).copy();schedule=pd.DataFrame(schedule if schedule is not None else games).copy()
    if df.empty:return df
    if schedule.empty:schedule=pd.DataFrame(columns=['away','home','season','week','kickoff','id'])
    for frame in (df,schedule):
        for a,b in [('away_team','away'),('home_team','home'),('game_id','id')]:
            if b not in frame and a in frame:frame[b]=frame[a]
        if not {'away','home'}.issubset(frame):raise ValueError('Games need away/home team codes.')
        if 'kickoff' not in frame and {'gameday','gametime'}.issubset(frame):
            frame['kickoff']=frame.gameday.astype(str)+' '+frame.gametime.astype(str)+' America/New_York'
        frame['_at']=frame.get('kickoff',pd.Series(index=frame.index,dtype=object)).map(utc)
        for side in ('away','home'):frame['_'+side]=frame[side].map(code)
    df['away_division']=df._away.map(DIVISIONS);df['home_division']=df._home.map(DIVISIONS)
    supported=numeric(df,'season').ge(2002)
    if 'league' in df:supported &= df.league.eq('NFL')
    df['divisional_matchup']=supported & df.away_division.notna() & df.away_division.eq(df.home_division)
    for side in ('away','home'):
        rest=[];byes=[];reasons=[]
        for _,game in df.iterrows():
            team=game['_'+side];season=game.get('season');week=game.get('week');at=game['_at']
            own=schedule[(schedule._away.eq(team)|schedule._home.eq(team)) & numeric(schedule,'season').eq(season)]
            if 'game_type' in own:own=own[own.game_type.eq('REG')]
            elif 'season_type' in own:own=own[own.season_type.isin(['REG','regular'])]
            prior=own[own._at.notna() & own._at.lt(at)].sort_values('_at') if pd.notna(at) else own.iloc[:0]
            supplied=pd.to_numeric(pd.Series([game.get(side+'_rest')]),errors='coerce').iloc[0]
            days=float(supplied) if pd.notna(supplied) and 0<supplied<=30 else None
            if days is None and not prior.empty:
                # Rest is measured by local game dates, not kickoff-hour differences.
                days=float((at.tz_convert('America/New_York').date()-prior.iloc[-1]._at.tz_convert('America/New_York').date()).days)
            explicit=game.get(side+'_off_bye')
            required=17 if pd.notna(season) and season>=2021 else 16
            complete=len(own.drop_duplicates('id' if 'id' in own else '_at'))>=required
            regular=game.get('game_type',game.get('season_type','REG')) in ('REG','regular')
            confirmed=bool(regular and complete and not prior.empty and pd.notna(week) and week>=3 and not numeric(own,'week').eq(week-1).any() and numeric(prior,'week').lt(week-1).all())
            is_explicit=isinstance(explicit,(bool,np.bool_))
            byes.append(bool(explicit) if is_explicit else confirmed)
            reasons.append('explicit bye flag' if is_explicit else 'complete schedule confirms previous-week bye' if confirmed else 'no confirmed previous-week bye')
            rest.append(days if days is not None and 0<days<=30 else np.nan)
        df[side+'_rest_days']=rest;df[side+'_off_bye']=byes;df[side+'_bye_source']=reasons
    df['rest_differential_home']=df.home_rest_days-df.away_rest_days
    return df.drop(columns=['_away','_home','_at'])
