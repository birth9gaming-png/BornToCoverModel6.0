"""Dated head-to-head context with recency weighting and small-sample shrinkage."""
from __future__ import annotations
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HISTORY_DEFAULTS=dict(h2h_margin_weight=.10,h2h_total_weight=.10,h2h_half_life_years=2.,h2h_prior_games=8.,h2h_difference_cap=10.,margin_scale=1.,margin_intercept=0.,total_scale=1.,total_intercept=0.)


def timestamp(value):
    if not value: raise ValueError('Missing timestamp')
    value=str(value)
    if value.endswith(' America/New_York'):
        return datetime.fromisoformat(value.replace(' America/New_York','')).replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def summarize(data,game,home_field):
    forecast=game.get('kickoff') or data.get('snapshot_date')
    try: at=timestamp(forecast)
    except ValueError: return dict(count=0,rows=[],status='Game date missing; history skipped.')
    par={**HISTORY_DEFAULTS,**data['parameters'][game['league']]}
    rows=[]; seen=set()
    for row in data.get('history',[]):
        if row.get('league')!=game['league'] or {row.get('away'),row.get('home')}!={game['away'],game['home']}: continue
        try:
            played=timestamp(row.get('kickoff'))
            a,h=float(row['away_score']),float(row['home_score'])
        except (ValueError,TypeError,KeyError): continue
        if played >= at or min(a,h)<0 or not all(math.isfinite(v) for v in (a,h)): continue
        identity=(row.get('id'),row['away'],row['home'],played.isoformat())
        if identity in seen: continue
        seen.add(identity)
        sign=1 if row['away']==game['away'] else -1
        margin=sign*(a-h)
        neutral_margin=margin + (0 if row.get('neutral') else sign*home_field)
        age=(at-played).total_seconds()/(365.25*86400)
        weight=2**(-age/par['h2h_half_life_years'])
        rows.append({**row,'current_away_margin':margin,'neutral_margin':neutral_margin,'game_total':a+h,'recency_weight':weight})
    rows.sort(key=lambda r:timestamp(r['kickoff']),reverse=True)
    if not rows: return dict(count=0,rows=[],status='No prior completed meetings available.')
    mass=sum(r['recency_weight'] for r in rows)
    return dict(count=len(rows),rows=rows,status='Prior completed meetings',wins_away=sum(r['current_away_margin']>0 for r in rows),ties=sum(r['current_away_margin']==0 for r in rows),weighted_games=mass,shrinkage=mass/(mass+par['h2h_prior_games']),mean_neutral_margin=sum(r['recency_weight']*r['neutral_margin'] for r in rows)/mass,mean_total=sum(r['recency_weight']*r['game_total'] for r in rows)/mass)


def adjustments(data,game,margin,total,venue,rest):
    par={**HISTORY_DEFAULTS,**data['parameters'][game['league']]}
    summary=summarize(data,game,data['parameters'][game['league']]['h'])
    if not summary['count']: return 0.,0.,summary
    # Remove this game's home/rest margin before comparing with neutral history.
    neutral=margin+venue+rest
    cap=par['h2h_difference_cap']
    clip=lambda x:max(-cap,min(cap,x))
    dm=par['h2h_margin_weight']*summary['shrinkage']*clip(summary['mean_neutral_margin']-neutral)
    dt=par['h2h_total_weight']*summary['shrinkage']*clip(summary['mean_total']-total)
    summary.update(margin_adjustment=dm,total_adjustment=dt)
    return dm,dt,summary
