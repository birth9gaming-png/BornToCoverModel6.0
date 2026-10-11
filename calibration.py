"""Fit score calibration only to timestamped, archived pregame forecasts."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from statistics import mean
from matchup_history import timestamp


def signature(data):
    p=copy.deepcopy(data['parameters'])
    for lg in ('NFL','CFB'):
        for key in ('margin_scale','margin_intercept','total_scale','total_intercept'):p[lg].pop(key,None)
    if data.get('cfb_boost_model'):
        p['cfb_fitted_model_hash']=hashlib.sha256(json.dumps(data['cfb_boost_model'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    if data.get('nfl_boost_model'):
        p['nfl_fitted_model_hash']=hashlib.sha256(json.dumps(data['nfl_boost_model'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    if data.get('nfl_layer_settings'):p['nfl_layer_settings']=copy.deepcopy(data['nfl_layer_settings'])
    if data.get('auto_injury_settings'):p['auto_injury_settings']=copy.deepcopy(data['auto_injury_settings'])
    if data.get('injury_player_values'):p['injury_player_values']=copy.deepcopy(data['injury_player_values'])
    return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()


def archive(data,results,now=None):
    now=now or datetime.now(timezone.utc); rows=[]
    for r in results:
        if 'away' not in r: continue
        g=r['game']
        try: kickoff=timestamp(g.get('kickoff'))
        except ValueError: continue
        if kickoff<=now or g.get('completed') or not g.get('id'):continue
        keys=('venue','rest','injury','automatic_injuries','weather','schedule_context','ats_history','nfl_efficiency_layer','nfl_dvoa_layer','nfl_trenches_layer','nfl_rest_layer','nfl_wind_layer','nfl_division_layer')
        mm=sum(r['away']['score_parts'].get(k,0)-r['home']['score_parts'].get(k,0) for k in keys)
        tm=sum(r['away']['score_parts'].get(k,0)+r['home']['score_parts'].get(k,0) for k in keys)
        rows.append(dict(id=g['id'],league=r['league'],away=g['away'],home=g['home'],kickoff=kickoff.isoformat(),created_at=now.isoformat(),model_signature=signature(data),raw_margin=r['raw_margin_away'],raw_total=r['raw_total'],margin_modifiers=mm,total_modifiers=tm,actual_away=None,actual_home=None))
    return dict(format='pregame-calibration-v1',forecasts=rows)


def attach_results(rows,data):
    outcomes={}
    for g in data.get('games',[]):
        if g.get('actual_away') is not None and g.get('actual_home') is not None:outcomes[(g.get('league'),str(g.get('id')))]=(g['actual_away'],g['actual_home'])
    for g in data.get('history',[]):outcomes[(g['league'],str(g['id']))]=(g['away_score'],g['home_score'])
    for g in data.get('schedule',[]):
        if g.get('completed') is True and g.get('away_score') is not None and g.get('home_score') is not None:
            outcomes[(g['league'],str(g['id']))]=(g['away_score'],g['home_score'])
    result=copy.deepcopy(rows)
    for row in result:
        if (row.get('league'),str(row.get('id'))) in outcomes:
            row['actual_away'],row['actual_home']=outcomes[(row['league'],str(row['id']))]
    return result


def candidate(rows,league,model_signature,now=None):
    now=now or datetime.now(timezone.utc); selected={};rejected=0
    for row in rows:
        if row.get('league')!=league:continue
        try:
            created,kickoff=timestamp(row['created_at']),timestamp(row['kickoff'])
            values={k:float(row[k]) for k in ('raw_margin','raw_total','margin_modifiers','total_modifiers','actual_away','actual_home')}
            if not all(math.isfinite(v) for v in values.values()) or min(values['actual_away'],values['actual_home'])<0 or not created<kickoff<now or row.get('model_signature')!=model_signature:
                raise ValueError('Not an eligible pregame forecast')
            key=str(row['id'])
            # One forecast per game; use the earliest archived one consistently.
            if key not in selected or created<timestamp(selected[key]['created_at']):selected[key]=row
        except (ValueError,TypeError,KeyError):rejected+=1
    ordered=sorted(selected.values(),key=lambda r:timestamp(r['kickoff']))
    if len(ordered)<80:raise ValueError(f'{league}: need at least 80 distinct completed games with matching archived pregame forecasts; found {len(ordered)}. Head-to-head scores alone cannot calibrate this model.')
    split=int(len(ordered)*.75)
    boundary=timestamp(ordered[split]['kickoff'])
    train=[r for r in ordered if timestamp(r['kickoff'])<boundary]
    holdout=[r for r in ordered if timestamp(r['kickoff'])>=boundary]
    if len(train)<50 or len(holdout)<20:raise ValueError('Need 50 training games and 20 later holdout games at distinct kickoff boundaries.')
    report=dict(league=league,training_games=len(train),holdout_games=len(holdout),rejected_rows=rejected,holdout_starts=boundary.isoformat(),coefficients={},metrics={})
    for target in ('margin','total'):
        def pair(r):
            mod=float(r[target+'_modifiers']);x=float(r['raw_'+target])-mod
            a,h=float(r['actual_away']),float(r['actual_home'])
            return x,(a-h if target=='margin' else a+h)-mod
        xy=[pair(r) for r in train];xm=mean(x for x,y in xy);ym=mean(y for x,y in xy)
        den=sum((x-xm)**2 for x,y in xy)
        if den<=1e-8:raise ValueError('Forecast variation is too small to fit ' + target)
        slope=sum((x-xm)*(y-ym) for x,y in xy)/den; intercept=ym-slope*xm
        if slope<=0 or not math.isfinite(slope):raise ValueError('Nonpositive calibration slope; review the feature model.')
        held=[pair(r) for r in holdout]
        base=math.sqrt(mean((x-y)**2 for x,y in held)); fitted=math.sqrt(mean((slope*x+intercept-y)**2 for x,y in held))
        report['coefficients'][target]=dict(scale=slope,intercept=intercept)
        report['metrics'][target]=dict(baseline_holdout_rmse=base,calibrated_holdout_rmse=fitted,improves=fitted<base)
    return report
