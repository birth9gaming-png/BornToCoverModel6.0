"""Serializable gradient-boosting corrections to the CFB statistical core."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np

FEATURES = ['ypp_gap','success_gap','explosiveness_gap','ypp_sum','success_sum','explosiveness_sum',
            'away_plays','home_plays','turnover_gap','baseline_neutral_margin','baseline_total',
            'massey_gap','colley_gap','away_sample_games','home_sample_games']
CORE_KEYS = ('by','bs','shrink','turnover_retention','turnover_point_value',
             'q','n','y','s','x','bx','h','power_weight','power_ridge','power_min_games','power_method','offense_share')


def signature(parameters):
    lp=parameters['CFB']
    values={key:lp.get(key,parameters.get(key)) for key in CORE_KEYS}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def features(data, game, result):
    a,h=(data['teams'][game[side]] for side in ('away','home'))
    ma,mh=result['away']['matched'],result['home']['matched']
    core_a=sum(result['away']['score_parts'].get(k,0) for k in ('pace_base','turnover'))
    core_h=sum(result['home']['score_parts'].get(k,0) for k in ('pace_base','turnover'))
    power=result.get('power_rating',{})
    margin=core_a-core_h
    if power.get('active'):
        raw_eff=result['away']['score_parts']['pace_base']-result['home']['score_parts']['pace_base']
        margin+=power['weight']*(power['neutral_margin']-raw_eff)
    def gap(key):
        av,hv=power.get(key+'_away'),power.get(key+'_home')
        if av is None or hv is None or (key=='massey' and not power.get('massey_connected')):return None
        return av-hv
    values=[ma['y']-mh['y'],ma['s']-mh['s'],ma['x']-mh['x'],ma['y']+mh['y'],ma['s']+mh['s'],ma['x']+mh['x'],
            a['p'],h['p'],a['t']-h['t'],margin,core_a+core_h,gap('massey'),gap('colley'),
            a.get('sample_games'),h.get('sample_games')]
    return dict(zip(FEATURES,values)),margin,core_a+core_h


def export_estimator(estimator):
    trees=[]
    for entry in estimator.estimators_.ravel():
        t=entry.tree_
        trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
                          feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,0].tolist()))
    return dict(initial=float(estimator.init_.constant_.ravel()[0]),rate=float(estimator.learning_rate),trees=trees)


def predict_estimator(estimator, vector):
    if not isinstance(estimator,dict) or not 0<float(estimator.get('rate',0))<=1 or len(estimator.get('trees',[]))>500:
        raise ValueError('Invalid gradient-boosting artifact.')
    vector=np.asarray(vector,dtype=np.float32); value=float(estimator['initial'])
    if not np.isfinite(vector).all():raise ValueError('Invalid prediction vector.')
    if not math.isfinite(value):raise ValueError('Invalid initial prediction.')
    for tree in estimator['trees']:
        n=len(tree['left'])
        if not 1<=n<=511 or any(len(tree[k])!=n for k in ('right','feature','threshold','value')):
            raise ValueError('Invalid tree structure.')
        node=0;seen=set()
        while True:
            if not isinstance(node,int) or not 0<=node<n or node in seen:raise ValueError('Invalid or cyclic tree.')
            seen.add(node)
            if tree['left'][node]==-1:
                leaf=float(tree['value'][node])
                if not math.isfinite(leaf):raise ValueError('Nonfinite leaf.')
                value+=estimator['rate']*leaf;break
            index=tree['feature'][node];threshold=float(tree['threshold'][node])
            if not isinstance(index,int) or not 0<=index<len(vector) or not math.isfinite(threshold):
                raise ValueError('Invalid split.')
            node=tree['left'][node] if vector[index]<=threshold else tree['right'][node]
    return value


def correction(model, feature_values):
    if model.get('format')!='cfb-boost-v1' or model.get('features')!=FEATURES or len(model.get('medians',[]))!=len(FEATURES):
        raise ValueError('Unsupported CFB model format.')
    medians=[float(v) for v in model['medians']]
    if not all(math.isfinite(v) for v in medians):raise ValueError('Invalid training medians.')
    vector=[float(feature_values[k]) if feature_values.get(k) is not None else medians[i] for i,k in enumerate(FEATURES)]
    if not all(math.isfinite(v) for v in vector):raise ValueError('Nonfinite forecast feature.')
    result={}
    for target in ('margin','total'):
        spec=model['targets'][target]
        if not isinstance(spec.get('enabled'),bool):raise ValueError('Model target enabled must be true or false.')
        cap=float(spec['cap'])
        if not 0<cap<=30:raise ValueError('Invalid correction cap.')
        delta=predict_estimator(spec['estimator'],vector) if spec.get('enabled') else 0.
        result[target]=max(-cap,min(cap,delta))
    return result


def adjustment(data, game, parameters, result):
    report=dict(active=False,margin_adjustment=0.,total_adjustment=0.,status='CFB gradient boosting needs uploaded historical data and a validated fit.')
    if game['league']!='CFB':
        report['status']='Gradient boosting is currently scoped to College Football.';return report
    model=data.get('cfb_boost_model')
    if not model:return report
    if not isinstance(model,dict):
        report['status']='Gradient boosting skipped: invalid model object.';return report
    if model.get('core_signature')!=signature(parameters):
        report['status']='Core settings changed since training; refit or restore the fitted settings to use gradient boosting.';return report
    weight=parameters['CFB'].get('ml_weight',1.)
    if weight==0:
        report['status']='Gradient-boosting blend is zero.';return report
    try:
        inputs,_,_=features(data,game,result);delta=correction(model,inputs)
        report.update(active=any(model['targets'][k].get('enabled') for k in ('margin','total')),
                      margin_adjustment=weight*delta['margin'],total_adjustment=weight*delta['total'],
                      status='CFB residual gradient boosting, selected on later validation seasons; optional injury/FEI/roster/context terms remain separate assumptions.')
    except (ValueError,TypeError,KeyError,IndexError) as exc:report['status']='Gradient boosting skipped: '+str(exc)
    return report
