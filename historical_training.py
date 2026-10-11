"""Prior-week CFB feature reconstruction and season-based model evaluation."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import median
import numpy as np
from model import ensure_cfb_defaults, project
from matchup_history import timestamp
from cfb_ml import FEATURES, features, signature, export_estimator


def dataset_hash(dataset):
    source={key:dataset[key] for key in ('games','box','advanced','lines')}
    return hashlib.sha256(json.dumps(source,sort_keys=True,allow_nan=False).encode()).hexdigest()


def build_rows(dataset, parameters, min_games=2, progress=None):
    """No target-game statistics, same-week outcomes, or end-season aggregates."""
    params=copy.deepcopy(parameters)
    for key in ('margin_scale','total_scale'):params['CFB'][key]=1.
    for key in ('margin_intercept','total_intercept'):params['CFB'][key]=0.
    stats={(r['id'],r['team']):r for r in dataset['box']}
    advanced={(r['id'],r['team']):r for r in dataset['advanced']}
    lines=defaultdict(list)
    for quote in dataset['lines']:lines[quote['id']].append(quote)
    now=datetime.now(timezone.utc)
    games=sorted([g for g in dataset['games'] if g.get('completed') and g.get('season_type')=='regular'
                  and g.get('neutral') is not None and g.get('away_score') is not None and g.get('home_score') is not None
                  and timestamp(g['kickoff'])<now],key=lambda g:timestamp(g['kickoff']))
    groups=defaultdict(list)
    for game in games:groups[(game['season'],game['week'])].append(game)
    rows=[];skipped=Counter()
    for i,((season,week),targets) in enumerate(sorted(groups.items())):
        # The earliest scheduled game, including games with missing scores,
        # fixes a shared pre-week feature cutoff.
        cutoff=min(timestamp(g['kickoff']) for g in dataset['games'] if g['season']==season and g['week']==week and g.get('season_type')=='regular')
        prior=[g for g in games if g['season']==season and g['week']<week and timestamp(g['kickoff'])<cutoff]
        histories=[dict(g,league='CFB',away='CFB:'+g['away'],home='CFB:'+g['home']) for g in prior]
        own_games=defaultdict(list)
        for past in prior:
            for side in ('away','home'):own_games[past[side]].append(past)
        teams={}
        for name,previous in own_games.items():
            if len(previous)<min_games:continue
            total=dict(yards=0.,plays=0.,allowed_yards=0.,allowed_plays=0.,net_to=0.,success=0.,def_success=0.,explosion=0.,def_explosion=0.,successful_plays=0.,def_successful_plays=0.,adv_plays=0.,def_adv_plays=0.)
            valid=True
            for past in previous:
                gid=past['id'];opp=past['home'] if past['away']==name else past['away']
                box,def_box,adv=stats.get((gid,name)),stats.get((gid,opp)),advanced.get((gid,name))
                if not box or not def_box or not adv or adv.get('season') not in (None,season) or adv.get('week') not in (None,past['week']):
                    valid=False;break
                total['yards']+=box['yards'];total['plays']+=box['plays']
                total['allowed_yards']+=def_box['yards'];total['allowed_plays']+=def_box['plays'];total['net_to']+=def_box['giveaways']-box['giveaways']
                for side,prefix in [('offense',''),('defense','def_')]:
                    val=adv[side];successes=val['plays']*val['successRate']
                    total[prefix+'success']+=successes
                    total[prefix+'adv_plays']+=val['plays']
                    total[prefix+'successful_plays']+=successes
                    total[prefix+'explosion']+=successes*val['explosiveness']
            if not valid or min(total[k] for k in ('plays','allowed_plays','adv_plays','def_adv_plays','successful_plays','def_successful_plays'))<=0:continue
            teams['CFB:'+name]=dict(league='CFB',name=name,y=total['yards']/total['plays'],dy=total['allowed_yards']/total['allowed_plays'],
                                   s=100*total['success']/total['adv_plays'],ds=100*total['def_success']/total['def_adv_plays'],
                                   x=total['explosion']/total['successful_plays'],dx=total['def_explosion']/total['def_successful_plays'],
                                   p=total['plays']/len(previous),t=total['net_to']/len(previous),sample_games=len(previous),feature_through_week=week-1)
        data=dict(parameters=params,teams=teams,games=[],season=season,rating_history=histories,history=[],schedule=[])
        ensure_cfb_defaults(data)
        for target in targets:
            game=dict(id=target['id'],league='CFB',away='CFB:'+target['away'],home='CFB:'+target['home'],neutral=target['neutral'],kickoff=target['kickoff'],
                      week=week,feature_through_week=week-1,line_away=None,total=None,quotes=[])
            try:
                result=project(data,game)
                if 'away' not in result:skipped['Missing or incomplete prior-week team statistics']+=1;continue
                vector,margin,total=features(data,game,result)
                line_quotes=[q for q in lines[target['id']] if q['away']==target['away'] and q['home']==target['home']]
                spreads=[-q['home_handicap'] for q in line_quotes if q['home_handicap'] is not None]
                totals=[q['total'] for q in line_quotes if q['total'] is not None]
                venue=0. if target['neutral'] else -params['CFB']['h']
                rows.append(dict(id=target['id'],season=season,week=week,kickoff=target['kickoff'],feature_cutoff=cutoff.isoformat(),
                                 away=target['away'],home=target['home'],features=vector,baseline_margin=margin+venue,baseline_total=total,venue_margin=venue,
                                 actual_margin=target['away_score']-target['home_score'],actual_total=target['away_score']+target['home_score'],
                                 line_away=median(spreads) if spreads else None,market_total=median(totals) if totals else None,
                                 origin='Retrospectively reconstructed from completed prior weeks; not an archived pregame forecast.'))
            except (ValueError,TypeError,KeyError) as exc:skipped[str(exc)]+=1
        if progress:progress((i+1)/max(1,len(groups)),f'Built season {season}, week {week}')
    return rows,dict(skipped)


def error_metrics(actual,predicted):
    delta=np.asarray(actual)-np.asarray(predicted)
    return dict(mae=float(np.mean(np.abs(delta))),rmse=float(np.sqrt(np.mean(delta**2))),bias=float(np.mean(delta)))


def fit_boost(rows, parameters, validation_start=2023, test_start=2025, source_hash='', league='CFB'):
    from sklearn.ensemble import GradientBoostingRegressor
    if league=='NFL':
        from nfl_ml import FEATURES as feature_names, signature as model_signature
    elif league=='CFB':
        feature_names,model_signature=FEATURES,signature
    else:raise ValueError('Training league must be NFL or CFB.')
    split={name:[r for r in rows if condition(r['season'])] for name,condition in [
        ('train',lambda year:year<validation_start),('validation',lambda year:validation_start<=year<test_start),('test',lambda year:year>=test_start)]}
    if validation_start>=test_start:raise ValueError('Validation must start before the test season.')
    if len(split['train'])<200 or len(split['validation'])<100 or len(split['test'])<100:
        raise ValueError('Need at least 200 earlier training games, 100 validation games and 100 later test games with complete prior-week inputs.')
    matrix=lambda values:np.array([[r['features'].get(k) if r['features'].get(k) is not None else np.nan for k in feature_names] for r in values],dtype=float)
    xtrain=matrix(split['train']);medians=[]
    for col in xtrain.T:medians.append(float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.)
    fill=lambda x:np.where(np.isfinite(x),x,np.asarray(medians))
    matrices={key:fill(matrix(values)) for key,values in split.items()}
    artifact=dict(format=league.lower()+'-boost-v1',features=feature_names,medians=medians,core_signature=model_signature(parameters),source_hash=source_hash,
                  validation_start=validation_start,test_start=test_start,targets={})
    report=dict(algorithm='GradientBoostingRegressor on statistical-core residuals',counts={k:len(v) for k,v in split.items()},
                league=league,validation_start=validation_start,test_start=test_start,targets={},source_hash=source_hash,core_signature=model_signature(parameters))
    forecasts={key:[dict(r) for r in value] for key,value in split.items()}
    for target,cap in [('margin',8. if league=='NFL' else 10.),('total',12. if league=='NFL' else 15.)]:
        base_key='baseline_'+target;actual_key='actual_'+target
        ytrain=np.array([r[actual_key]-r[base_key] for r in split['train']])
        vactual=np.array([r[actual_key] for r in split['validation']]);vbase=np.array([r[base_key] for r in split['validation']])
        candidates=[]
        for depth,trees in [(2,80),(2,160),(3,80),(3,160)]:
            estimator=GradientBoostingRegressor(loss='squared_error',learning_rate=.04,n_estimators=trees,max_depth=depth,min_samples_leaf=20,random_state=7)
            estimator.fit(matrices['train'],ytrain)
            prediction=vbase+np.clip(estimator.predict(matrices['validation']),-cap,cap)
            candidates.append((error_metrics(vactual,prediction)['rmse'],depth,trees,estimator))
        _,depth,trees,estimator=min(candidates,key=lambda value:value[:3])
        baseline_validation=error_metrics(vactual,vbase)
        improved=candidates and min(v[0] for v in candidates)<baseline_validation['rmse']
        artifact['targets'][target]=dict(enabled=bool(improved),cap=cap,estimator=export_estimator(estimator))
        metrics={}
        # Test outcomes are first evaluated after the candidate is selected
        # using validation; no test statistic changes the chosen estimator.
        for key,values in split.items():
            actual=np.array([r[actual_key] for r in values]);base=np.array([r[base_key] for r in values])
            predicted=base+np.clip(estimator.predict(matrices[key]),-cap,cap) if improved else base
            metrics[key]=dict(baseline=error_metrics(actual,base),selected=error_metrics(actual,predicted))
            for r,pred in zip(forecasts[key],predicted):r['selected_'+target]=float(pred)
        residual=vactual-(vbase+np.clip(estimator.predict(matrices['validation']),-cap,cap) if improved else vbase)
        sigma=max(1.,float(np.std(residual,ddof=1)))
        report['targets'][target]=dict(enabled=bool(improved),depth=depth,trees=trees,correction_cap=cap,validation_sigma=sigma,metrics=metrics)
    report['market_test']={}
    test=forecasts['test']
    for target,market in [('margin','line_away'),('total','market_total')]:
        available=[r for r in test if r[market] is not None]
        if available:
            actual=[r['actual_'+target] for r in available]
            market_predictions=[-r[market] if target=='margin' else r[market] for r in available]
            report['market_test'][target]=dict(games=len(available),historical_market=error_metrics(actual,market_predictions),
                                              model_same_games=error_metrics(actual,[r['selected_'+target] for r in available]))
    report['probability_test']=probability_metrics(test,report)
    return artifact,report,forecasts


def probability_metrics(rows,report):
    """Score whole/half-point references and record excluded-line coverage.

    Book medians can fall between the supported market increments. Keep those
    games in fitting and point-error benchmarks, but do not round their lines
    or pass them to the live quote probability function.
    """
    from model import probabilities
    output={};coverage={}
    for target,line_key in [('margin','line_away'),('total','market_total')]:
        counts=dict(test_games=len(rows),scored_games=0,missing_lines=0,
                    invalid_lines=0,non_half_point_lines=0)
        available=[]
        for row in rows:
            line=row.get(line_key)
            if line is None:counts['missing_lines']+=1;continue
            try:
                if isinstance(line,bool):raise ValueError('Boolean market line')
                threshold=-float(line) if target=='margin' else float(line)
            except (ValueError,TypeError,OverflowError):
                counts['invalid_lines']+=1;continue
            doubled=2*threshold
            if not math.isfinite(threshold) or not math.isfinite(doubled):
                counts['invalid_lines']+=1;continue
            if not math.isclose(doubled,round(doubled)):
                counts['non_half_point_lines']+=1;continue
            available.append((row,threshold))
        counts['scored_games']=len(available);coverage[target]=counts
        if not available:continue
        sigma=float(report['targets'][target]['validation_sigma']);values=[]
        if not math.isfinite(sigma) or sigma<=0:
            raise ValueError('Validation uncertainty must be finite and positive for probability calibration.')
        for row,threshold in available:
            p=probabilities(row['selected_'+target],sigma,threshold,True)
            actual=row['actual_'+target];category=0 if actual>threshold else 1 if actual==threshold else 2
            onehot=[float(i==category) for i in range(3)]
            values.append(dict(p_win=p[0],win=onehot[0],brier=sum((a-b)**2 for a,b in zip(p,onehot)),
                               log_loss=-float(np.log(max(1e-12,p[category])))))
        bins=[]
        for index in range(10):
            lo=index/10
            group=[v for v in values if min(9,int(10*v['p_win']))==index]
            if group:bins.append(dict(probability_range=f'{lo:.1f}–{lo+.1:.1f}',games=len(group),predicted_win_rate=float(np.mean([v['p_win'] for v in group])),observed_win_rate=float(np.mean([v['win'] for v in group]))))
        output[target]=dict(games=len(values),three_outcome_brier=float(np.mean([v['brier'] for v in values])),
                            log_loss=float(np.mean([v['log_loss'] for v in values])),validation_sigma=sigma,reliability=bins)
    report['probability_test_coverage']=coverage
    return output
