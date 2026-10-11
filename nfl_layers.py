"""NFL-only wrappers around model.project/run. The core architecture is unchanged.

The defaults below are user-requested scenario priors, not fitted coefficients.
All layers are ON; missing observed inputs skip just the unavailable term.
"""
from __future__ import annotations
import copy
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from model import project as core_project, evaluate, quotes_for
from nfl_features_ext import engineer_game_context,utc
from nfl_stats import code

DEFAULTS=dict(enabled=True,efficiency_enabled=True,dvoa_enabled=True,trenches_enabled=True,
    weather_enabled=True,rest_enabled=True,division_enabled=True,underdog_enabled=True,
    attd_enabled=True,attd_recency_enabled=True,attd_redzone_enabled=True,attd_xtd_enabled=True,
    attd_defense_enabled=True,fetch_player_data=True,fetch_pressure_data=True,fetch_attd_odds=True,fetch_weather_data=True,
    pass_epa_weight=2.5,rush_epa_weight=1.,efficiency_blend=.25,success_mix=.25,
    success_prior_plays=100.,success_points_per_rate=25.,efficiency_cap=5.,
    dvoa_early_weight=.35,dvoa_late_weight=.15,dvoa_cap=3.,
    pressure_reference=.30,pressure_points=10.,line_yards_reference=4.3,line_yards_points=.75,trenches_cap=3.,
    wind_threshold=15.,wind_pass_loss_at_threshold=.05,wind_pass_loss_per_mph=.01,
    wind_pass_loss_cap=.30,wind_total_at_threshold=1.5,wind_total_per_mph=.30,wind_total_cap=8.,
    rest_points_per_day=.12,rest_cap=1.5,bye_bonus=.6,
    division_margin_shrink=.12,division_total_drop=1.5,division_dog_priority=.01,
    short_home_dog_max=3.5,short_home_dog_priority=.015,
    attd_min_total=47.5,attd_window=4.,attd_min_rb_volume=10.,attd_min_wr_volume=5.,attd_min_te_volume=4.,
    attd_bounce_bonus=.03,attd_sell_multiplier=.88,attd_redzone_weight=.25,attd_xtd_weight=.25,
    attd_defense_weight=.35,attd_probability_cap=.95)
BOOL_KEYS={k for k,v in DEFAULTS.items() if isinstance(v,bool)}
SCORE_KEYS=('nfl_efficiency_layer','nfl_dvoa_layer','nfl_trenches_layer','nfl_rest_layer','nfl_wind_layer','nfl_division_layer')
TEAM_FIELDS=('offensive_dvoa','defensive_dvoa','total_dvoa','adjusted_line_yards','defensive_adjusted_line_yards','non_blitz_pressure_rate','redzone_defensive_dvoa')


def settings(data=None):
    config={**DEFAULTS,**((data or {}).get('nfl_layer_settings') or {})}
    for key,default in DEFAULTS.items():
        value=config[key]
        if key in BOOL_KEYS:
            if not isinstance(value,bool):raise ValueError('NFL toggle must be Boolean: '+key)
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('NFL layer weight must be finite and nonnegative: '+key)
    for key in ('efficiency_blend','success_mix','dvoa_early_weight','dvoa_late_weight','wind_pass_loss_at_threshold','wind_pass_loss_per_mph','wind_pass_loss_cap','division_margin_shrink','attd_bounce_bonus','attd_sell_multiplier','attd_probability_cap','pressure_reference'):
        if config[key]>1:raise ValueError('NFL fraction must be 0–1: '+key)
    if config['pass_epa_weight']+config['rush_epa_weight']<=0:raise ValueError('At least one EPA weight must be positive.')
    if not 1<=config['attd_window']<=20 or config['attd_window']!=int(config['attd_window']):raise ValueError('ATTD lookback must be 1–20 active games.')
    return config


def forecast_as_of(data):
    for key in ('as_of','fetched_at'):
        at=utc(data.get(key))
        if pd.notna(at):return at
    date=data.get('snapshot_date')
    if date:
        at=utc(str(date)+' 23:59:59 America/New_York')
        if pd.notna(at):return min(at,pd.Timestamp(datetime.now(timezone.utc)))
    return pd.Timestamp(datetime.now(timezone.utc))


def eligible_team_metrics(data,game,team_id):
    """Import snapshots must precede both the forecast cutoff and kickoff."""
    cutoff=game.get('feature_through_week',game.get('week',1)-1)
    season=game.get('season',data.get('season'))
    at=forecast_as_of(data);kickoff=utc(game.get('kickoff'))
    if pd.notna(kickoff):at=min(at,kickoff)
    rows=[]
    for row in data.get('nfl_team_inputs',[]):
        published=utc(row.get('as_of'))
        if code(row.get('team',''))==code(team_id) and row.get('season')==season and row.get('through_week',999)<=cutoff and pd.notna(published) and published<=at:
            rows.append(row)
    return max(rows,key=lambda r:(r['through_week'],utc(r['as_of'])),default={})


def apply_divisional_modifiers(predictions, margin_shrink=.12, total_drop=1.5):
    """Pandas override: margin is AWAY minus HOME; spread is its negative."""
    if not 0<=margin_shrink<=1 or total_drop<0:raise ValueError('Invalid divisional modifiers.')
    result=predictions.copy()
    mask=result['divisional_matchup'].fillna(False).astype(bool)
    result['margin_away']=np.where(mask,result.margin_away*(1-margin_shrink),result.margin_away)
    result['fair_total']=np.where(mask,result.fair_total-total_drop,result.fair_total)
    result['fair_total']=np.maximum(result.fair_total,np.abs(result.margin_away)+1.)
    result['fair_line_away']=-result.margin_away
    return result


def _finite(value):
    return isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value)


def _cap(value,limit):return float(np.clip(value,-limit,limit))


def apply_nfl_layers(data,game,result,config=None):
    """Apply additive NFL evidence, then rivalry shrinkage, and recompute EV.

    Underdog preference changes selection_score, never p_win or quoted EV.
    It is distinct from the probability change caused by spread compression.
    """
    config=config or settings(data)
    if result.get('league')!='NFL' or 'away' not in result or not config['enabled']:return result
    r=copy.deepcopy(result);par=data['parameters']['NFL']
    g={**game,'season':game.get('season',data.get('season',int(str(data.get('snapshot_date','2026'))[:4])))}
    context=engineer_game_context([g],data.get('schedule',[])).iloc[0].to_dict()
    imported={side:eligible_team_metrics(data,g,game[side]) for side in ('away','home')}
    teams={side:{**data['teams'][game[side]],**{k:v for k,v in imported[side].items() if k in TEAM_FIELDS}} for side in ('away','home')}
    report=dict(active=True,context=context,terms={},notes=[],parameters=copy.deepcopy(config),
                base_margin_away=r['margin_away'],base_total=r['fair_total'],team_features={})
    pace=r.get('pace',1.);n=par['n']*pace;q=par['q']*pace
    delta={side:dict.fromkeys(SCORE_KEYS,0.) for side in ('away','home')}
    wp,wr=config['pass_epa_weight'],config['rush_epa_weight']
    for side,other in [('away','home'),('home','away')]:
        own,opp=teams[side],teams[other]
        def matched(off,de):
            a,b=own.get(off),opp.get(de)
            return (a+b)/2 if all(_finite(v) for v in (a,b)) else None
        passing=matched('pass_epa','dpass_epa');rushing=matched('rush_epa','drush_epa')
        volume=own.get('sample_plays',0) or 0
        success=own.get('s');allowed=opp.get('ds')
        smoothed=None
        if all(_finite(v) for v in (success,allowed,volume)):
            k=config['success_prior_plays'];prior=par['s']/100
            own_sr=(volume*success/100+k*prior)/(volume+k) if volume+k>0 else prior
            ov=opp.get('sample_plays',0) or 0
            allowed_sr=(ov*allowed/100+k*prior)/(ov+k) if ov+k>0 else prior
            smoothed=(own_sr+allowed_sr)/2
        weighted_epa=(wp*passing+wr*rushing)/(wp+wr) if passing is not None and rushing is not None else None
        features=dict(matched_pass_epa=passing,matched_rush_epa=rushing,weighted_epa=weighted_epa,
                      smoothed_matched_success=smoothed,**{k:own.get(k) for k in TEAM_FIELDS})
        report['team_features'][side]=features
        if config['efficiency_enabled']:
            if weighted_epa is not None:
                target=q+n*(weighted_epa-par.get('epa_reference',0.))
                if smoothed is not None:
                    sr_score=q+pace*config['success_points_per_rate']*(smoothed-par['s']/100)
                    target=(1-config['success_mix'])*target+config['success_mix']*sr_score
                base=sum(r[side]['score_parts'].get(k,0.) for k in ('pace_base','nfl_epa'))
                delta[side]['nfl_efficiency_layer']=_cap(config['efficiency_blend']*(target-base),config['efficiency_cap'])
            else:report['notes'].append(side+': passing/rushing EPA incomplete; efficiency layer skipped.')
        if config['dvoa_enabled']:
            off,de=own.get('offensive_dvoa'),opp.get('defensive_dvoa')
            if _finite(off) and _finite(de):
                week=float(g.get('week') or (g.get('feature_through_week',0)+1))
                late=float(np.clip((week-8)/5,0,1))
                weight=config['dvoa_early_weight']*(1-late)+config['dvoa_late_weight']*late
                factor=1+float(np.clip((off+de)/200,-.35,.35))
                delta[side]['nfl_dvoa_layer']=_cap(weight*q*(factor-1),config['dvoa_cap'])
                features.update(dvoa_baseline_factor=factor,dvoa_weight=weight)
            else:report['notes'].append(side+': dated offensive/opposing defensive DVOA missing; DVOA term skipped.')
        if config['trenches_enabled']:
            trench=0.;pressure=opp.get('non_blitz_pressure_rate')
            if _finite(pressure):trench-=config['pressure_points']*(pressure-config['pressure_reference'])*wp/(wp+wr)
            else:report['notes'].append(side+': opposing non-blitz pressure missing; pressure term skipped.')
            aly=own.get('adjusted_line_yards');allowed_aly=opp.get('defensive_adjusted_line_yards')
            source='official adjusted line yards'
            if not _finite(aly):aly=own.get('line_yards_proxy');source='unadjusted PBP line-yards proxy'
            if not _finite(allowed_aly):allowed_aly=opp.get('defensive_line_yards_proxy')
            if _finite(aly):
                line=(aly+allowed_aly)/2 if _finite(allowed_aly) else aly
                trench+=config['line_yards_points']*(line-config['line_yards_reference'])*wr/(wp+wr)
                features['line_yards_used']=line;features['line_yards_source']=source
            else:report['notes'].append(side+': line yards unavailable; run-blocking term skipped.')
            delta[side]['nfl_trenches_layer']=_cap(pace*trench,config['trenches_cap'])
    if config['rest_enabled']:
        difference=context['rest_differential_home']
        target=_cap(config['rest_points_per_day']*difference,config['rest_cap']) if _finite(difference) else 0.
        if not _finite(difference):report['notes'].append('Rest days incomplete; day-difference term skipped.')
        target+=config['bye_bonus']*(int(context['home_off_bye'])-int(context['away_off_bye']))
        existing=r.get('nfl_extensions',{}).get('rest_margin_home',0.)
        additional=target-existing
        delta['away']['nfl_rest_layer']=-additional/2;delta['home']['nfl_rest_layer']=additional/2
        report['rest_margin_home']=target
    if config['weather_enabled']:
        candidates=[g]+[w for w in data.get('nfl_weather_inputs',[]) if str(w.get('game_id'))==str(g.get('id'))]
        cutoff=forecast_as_of(data);kickoff=utc(g.get('kickoff'))
        if pd.notna(kickoff):cutoff=min(cutoff,kickoff)
        winds=[w for w in candidates if _finite(w.get('forecast_wind_mph')) and 0<=w['forecast_wind_mph']<=150 and pd.notna(utc(w.get('weather_as_of'))) and utc(w.get('weather_as_of'))<=cutoff]
        weather=max(winds,key=lambda w:utc(w['weather_as_of']),default=None)
        roof=str((weather or {}).get('roof',g.get('roof','unknown'))).lower()
        if weather and roof in ('outdoors','open','outdoor'):
            wind=weather['forecast_wind_mph']
            if wind>=config['wind_threshold']:
                over=wind-config['wind_threshold']
                loss=min(config['wind_pass_loss_cap'],config['wind_pass_loss_at_threshold']+over*config['wind_pass_loss_per_mph'])
                drop=min(config['wind_total_cap'],config['wind_total_at_threshold']+over*config['wind_total_per_mph'])
                # Replace any existing manual game weather adjustment to avoid doubling it.
                additional=-drop-float(g.get('weather_total') or 0.)
                for side in ('away','home'):
                    passing=report['team_features'][side]['matched_pass_epa']
                    report['team_features'][side]['wind_adjusted_pass_epa']=passing-loss*max(passing,.10) if passing is not None else None
                    delta[side]['nfl_wind_layer']=additional/2
                report.update(wind_mph=wind,passing_loss_fraction=loss,weather_total_effect=-drop)
        elif not weather:report['notes'].append('Dated sustained-wind forecast unavailable; wind term skipped.')
        elif roof not in ('dome','closed','indoors'):report['notes'].append('Roof exposure unknown; wind term skipped.')
    # Familiarity adjustment is applied to the FINAL pre-division score pair.
    margin=r['margin_away']+sum(delta['away'].values())-sum(delta['home'].values())
    total=r['fair_total']+sum(delta['away'].values())+sum(delta['home'].values())
    if config['division_enabled'] and context['divisional_matchup']:
        transformed=apply_divisional_modifiers(pd.DataFrame([dict(margin_away=margin,fair_total=total,divisional_matchup=True)]),config['division_margin_shrink'],config['division_total_drop']).iloc[0]
        dm,dt=transformed.margin_away-margin,transformed.fair_total-total
        delta['away']['nfl_division_layer']=(dm+dt)/2;delta['home']['nfl_division_layer']=(dt-dm)/2
    for side in ('away','home'):
        r[side]['score_parts'].update(delta[side]);r[side]['score']=sum(r[side]['score_parts'].values())
        if r[side]['score']<=0:raise ValueError('NFL layer settings produce a nonpositive score; reduce the adjustments.')
    r['margin_away']=r['away']['score']-r['home']['score'];r['fair_total']=r['away']['score']+r['home']['score'];r['fair_line_away']=-r['margin_away']
    dm,dt=r['margin_away']-report['base_margin_away'],r['fair_total']-report['base_total']
    r['raw_margin_away']+=dm;r['raw_total']+=dt
    r['spread_edge_away']=r['margin_away']+game['line_away'] if game.get('line_away') is not None else None
    r['total_edge_over']=r['fair_total']-game['total'] if game.get('total') is not None else None
    report['terms']={key:dict(away=delta['away'][key],home=delta['home'][key]) for key in SCORE_KEYS}
    report.update(margin_adjustment=dm,total_adjustment=dt);r['nfl_layers']=report
    r['markets']=[];r['quote_errors']=[]
    for quote in quotes_for(game):
        try:
            item=evaluate(quote,r,par);bonus=0.
            if config['underdog_enabled'] and context['divisional_matchup'] and item['kind']=='spread' and item['line']>0:
                bonus=config['division_dog_priority']
                if item['selection']=='home' and not game.get('neutral') and item['line']<=config['short_home_dog_max']:bonus+=config['short_home_dog_priority']
            item.update(ats_priority_bonus=bonus,selection_score=item['conditional_ev']+bonus)
            r['markets'].append(item)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:r['quote_errors'].append(str(exc))
    r['best_market']=max(r['markets'],key=lambda v:v['selection_score'],default=None)
    return r


def project(data,game,parameters=None):
    result=core_project(data,game,parameters)
    if result.get('league')!='NFL':return result
    return apply_nfl_layers({**data,'parameters':parameters or data['parameters']},game,result)


def run(data,parameters=None):
    results=[]
    for game in data['games']:
        try:results.append(project(data,game,parameters))
        except (ValueError,ArithmeticError,TypeError,KeyError) as exc:
            results.append(dict(game=copy.deepcopy(game),league=game.get('league'),status=str(exc),markets=[],best_market=None))
    return results
