"""Regularized opponent-adjusted margin ratings from prior completed games."""
from __future__ import annotations
from functools import lru_cache
from collections import Counter
import numpy as np
from matchup_history import timestamp
from cfb_ratings import fit as cfb_fit


def defaults(league):
    # Editable starting assumptions, not historically selected coefficients.
    return dict(power_weight=.50 if league=='CFB' else .25,power_ridge=2.,power_min_games=2.)


@lru_cache(maxsize=32)
def _fit(samples, ridge, home_field):
    names=sorted({name for a,h,*_ in samples for name in (a,h)})
    index={name:i for i,name in enumerate(names)}
    design=np.zeros((len(samples),len(names))); margins=np.zeros(len(samples)); counts=dict.fromkeys(names,0)
    for i,(away,home,away_score,home_score,neutral) in enumerate(samples):
        design[i,index[away]]=1.;design[i,index[home]]=-1.
        margins[i]=away_score-home_score+(0. if neutral else home_field)
        counts[away]+=1;counts[home]+=1
    # min_r ||Xr-y||² + ridge*||r||². Positive ridge also anchors the
    # league mean at zero and identifies disconnected early-season schedules.
    ratings=np.linalg.solve(design.T@design+ridge*np.eye(len(names)),design.T@margins)
    return dict(zip(names,ratings.tolist())),counts


def comparison(data, game, params, efficiency_margin):
    league=game['league']; par={**defaults(league),**params[league]}
    result=dict(active=False,weight=par['power_weight'],margin_adjustment=0.,neutral_margin=None,away_rating=None,home_rating=None,prior_games=0,status='No prior result history; refresh the slate to load opponent-adjusted ratings.')
    if par['power_weight']==0 and league!='CFB':
        result['status']='Opponent-adjusted result weight is zero.';return result
    try: cutoff=timestamp(game.get('kickoff'))
    except ValueError:
        result['status']='Game kickoff missing; result ratings skipped.';return result
    samples=[];seen=set()
    for row in data.get('rating_history',[]):
        if row.get('league')!=league or row.get('completed') is False:continue
        if data.get('season') is not None and row.get('season')!=data['season']:continue
        week=game.get('feature_through_week')
        if week is not None and (row.get('week') is None or row['week']>week):continue
        try:
            played=timestamp(row.get('kickoff'));a=float(row['away_score']);h=float(row['home_score'])
        except (ValueError,TypeError,KeyError):continue
        if played>=cutoff or min(a,h)<0 or not np.isfinite((a,h)).all():continue
        identity=(row.get('id'),row.get('away'),row.get('home'),played.isoformat())
        if identity in seen or not row.get('away') or not row.get('home') or row['away']==row['home']:continue
        seen.add(identity);samples.append((row['away'],row['home'],a,h,bool(row.get('neutral'))))
    if not samples:return result
    samples=tuple(sorted(samples))
    counts=Counter(name for row in samples for name in row[:2])
    result.update(prior_games=len(samples),away_games=counts.get(game['away'],0),home_games=counts.get(game['home'],0))
    ratings={}
    if league!='CFB' or par.get('power_method','massey')=='ridge':
        ratings,counts=_fit(samples,par['power_ridge'],params[league]['h'])
    if league=='CFB':
        extra=cfb_fit(samples,params[league]['h'])
        result.update(colley_away=extra['colley'].get(game['away']),colley_home=extra['colley'].get(game['home']),
                      massey_away=extra['massey'].get(game['away']),massey_home=extra['massey'].get(game['home']))
        connected=game['away'] in extra['components'] and game['home'] in extra['components'] and extra['components'][game['away']]==extra['components'][game['home']]
        result['massey_connected']=connected
        if par.get('power_method','massey')=='massey':
            if not connected:
                result['status']='Prior schedules are disconnected; Massey point comparison withheld. Colley rankings remain descriptive.';return result
            ratings=extra['massey']
        result['method']=par.get('power_method','massey')
    result.update(prior_games=len(samples),away_games=counts.get(game['away'],0),home_games=counts.get(game['home'],0))
    if par['power_weight']==0:
        result['status']='Point-rating blend is zero; Colley and Massey ratings remain available as CFB training inputs.';return result
    if min(result['away_games'],result['home_games'])<par['power_min_games']:
        result['status']='Too few prior results for one or both teams; result ratings skipped.';return result
    away=ratings[game['away']];home=ratings[game['home']];neutral=away-home
    result.update(active=True,away_rating=away,home_rating=home,neutral_margin=neutral,margin_adjustment=par['power_weight']*(neutral-efficiency_margin),status='Prior completed results, adjusted for opponents and venue; blend weight is provisional. '+('Colley ranks wins/losses; Massey measures point strength.' if league=='CFB' else ''))
    return result
