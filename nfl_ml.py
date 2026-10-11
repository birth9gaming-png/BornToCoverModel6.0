"""Independent, safe JSON NFL gradient-boosting predictions and optional scenarios."""
from __future__ import annotations
import hashlib
import json
import math
from cfb_ml import CORE_KEYS, export_estimator, predict_estimator
from nfl_stats import rest_days

DEFAULTS=dict(ml_weight=1.,epa_weight=0.,epa_reference=0.,rest_day_weight=0.,rest_cap_points=1.5)
FEATURES=['ypp_gap','success_gap','explosiveness_gap','ypp_sum','success_sum','explosiveness_sum',
          'away_plays','home_plays','turnover_gap','baseline_margin','baseline_total','result_rating_gap',
          'epa_gap','epa_sum','pass_epa_gap','pass_epa_sum','rush_epa_gap','rush_epa_sum',
          'cpoe_gap','sack_rate_gap','pressure_sack_gap','away_rest_days','home_rest_days',
          'away_sample_games','home_sample_games','neutral_site']


def signature(parameters):
    par={**DEFAULTS,**parameters['NFL']}
    keys=CORE_KEYS+('epa_weight','epa_reference','rest_day_weight','rest_cap_points')
    values={k:par.get(k,parameters.get(k)) for k in keys}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def scenarios(game,away,home,par,pace):
    rest=rest_days(game)
    rest_margin=0.
    if all(v is not None for v in rest.values()):
        rest_margin=max(-par['rest_cap_points'],min(par['rest_cap_points'],par['rest_day_weight']*(rest['home']-rest['away'])))
    scores={};active=all(t.get(k) is not None for t in (away,home) for k in ('epa','depa'))
    for side,own,opp in [('away',away,home),('home',home,away)]:
        scores[side]=pace*par['q']+pace*par['n']*((own['epa']+opp['depa'])/2-par['epa_reference']) if active else None
    return dict(epa_available=active,epa_weight=par['epa_weight'] if active else 0.,epa_scores=scores,
        rest_days=rest,rest_margin_home=rest_margin,
        status='EPA and rest are prior-week inputs to the trained model. Optional direct weights start at zero to avoid counting the same efficiency twice.')


def features(data,game,result):
    away,home=(data['teams'][game[s]] for s in ('away','home'))
    ma,mh=result['away']['matched'],result['home']['matched']
    core=lambda side:sum(result[side]['score_parts'].get(k,0) for k in ('pace_base','turnover','nfl_epa','power_rating'))
    a,h=core('away'),core('home')
    extra=result.get('nfl_extensions',{})
    venue=0. if game.get('neutral') else -data['parameters']['NFL']['h']
    margin=a-h+venue-extra.get('rest_margin_home',0.)
    total=a+h
    def matched(key,defkey):
        av,bv=away.get(key),home.get(defkey)
        cv,dv=home.get(key),away.get(defkey)
        return ((av+bv)/2,(cv+dv)/2) if all(v is not None for v in (av,bv,cv,dv)) else (None,None)
    values=dict(ypp_gap=ma['y']-mh['y'],success_gap=ma['s']-mh['s'],explosiveness_gap=ma['x']-mh['x'],
        ypp_sum=ma['y']+mh['y'],success_sum=ma['s']+mh['s'],explosiveness_sum=ma['x']+mh['x'],
        away_plays=away['p'],home_plays=home['p'],turnover_gap=away['t']-home['t'],
        baseline_margin=margin,baseline_total=total,result_rating_gap=result.get('power_rating',{}).get('neutral_margin'),
        away_rest_days=rest_days(game)['away'],home_rest_days=rest_days(game)['home'],
        away_sample_games=away.get('sample_games'),home_sample_games=home.get('sample_games'),neutral_site=float(bool(game.get('neutral'))))
    for name,off,defkey in [('epa','epa','depa'),('pass_epa','pass_epa','dpass_epa'),('rush_epa','rush_epa','drush_epa')]:
        x,y=matched(off,defkey);values[name+'_gap']=x-y if x is not None else None;values[name+'_sum']=x+y if x is not None else None
    for name,key in [('cpoe_gap','cpoe'),('sack_rate_gap','sack_rate'),('pressure_sack_gap','dsack_rate')]:
        av,hv=away.get(key),home.get(key);values[name]=av-hv if av is not None and hv is not None else None
    return values,margin,total


def correction(model,feature_values):
    if model.get('format')!='nfl-boost-v1' or model.get('features')!=FEATURES or len(model.get('medians',[]))!=len(FEATURES):
        raise ValueError('Unsupported NFL model format.')
    medians=[float(v) for v in model['medians']]
    if not all(math.isfinite(v) for v in medians):raise ValueError('Invalid training medians.')
    vector=[float(feature_values[k]) if feature_values.get(k) is not None else medians[i] for i,k in enumerate(FEATURES)]
    if not all(math.isfinite(v) for v in vector):raise ValueError('Nonfinite NFL features.')
    output={}
    for target in ('margin','total'):
        spec=model['targets'][target]
        if not isinstance(spec.get('enabled'),bool):raise ValueError('Invalid target enable flag.')
        cap=float(spec['cap'])
        if not 0<cap<=30:raise ValueError('Invalid correction cap.')
        value=predict_estimator(spec['estimator'],vector) if spec['enabled'] else 0.
        output[target]=max(-cap,min(cap,value))
    return output


def adjustment(data,game,parameters,result):
    report=dict(active=False,margin_adjustment=0.,total_adjustment=0.,status='NFL gradient boosting needs historical data and a validated fit.')
    model=data.get('nfl_boost_model')
    if not model:return report
    if not isinstance(model,dict):report['status']='Invalid NFL model object.';return report
    if any(data['teams'][game[side]].get(k) is None for side in ('away','home') for k in ('epa','depa')):
        report['status']='Refresh NFL statistics to provide EPA before applying this trained model.';return report
    if model.get('core_signature')!=signature(parameters):
        report['status']='NFL core settings changed; refit or restore the trained settings.';return report
    weight=parameters['NFL'].get('ml_weight',1.)
    if weight==0:report['status']='NFL gradient-boosting blend is zero.';return report
    try:
        inputs,_,_=features({**data,'parameters':parameters},game,result);delta=correction(model,inputs)
        report.update(active=any(model['targets'][k]['enabled'] for k in ('margin','total')),
            margin_adjustment=weight*delta['margin'],total_adjustment=weight*delta['total'],
            status='NFL residual gradient boosting, selected on validation seasons; manual injuries, weather, ATS and matchup history remain separate assumptions.')
    except (ValueError,TypeError,KeyError,IndexError) as exc:report['status']='NFL model skipped: '+str(exc)
    return report
