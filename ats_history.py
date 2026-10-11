"""Venue-specific ten-year ATS records with conservative point adjustments."""
from __future__ import annotations
import math
from collections import defaultdict
from zoneinfo import ZoneInfo
from matchup_history import timestamp

DEFAULTS=dict(ats_weight=2.,ats_prior_games=40.,ats_cap_points=.5)


def attach_records(teams, rows, cutoff, source, missing_years=()):
    end=timestamp(cutoff)
    local_end=end.astimezone(ZoneInfo('America/New_York'))
    try: begin=local_end.replace(year=local_end.year-10)
    except ValueError: begin=local_end.replace(year=local_end.year-10,day=28)
    begin=begin.astimezone(end.tzinfo)
    records=defaultdict(lambda:{venue:dict(wins=0,losses=0,pushes=0,missing_lines=0,unknown_venues=0) for venue in ('home','away')})
    seen=set()
    for row in rows:
        try: played=timestamp(row.get('kickoff'));a=float(row['away_score']);h=float(row['home_score'])
        except (ValueError,TypeError,KeyError):continue
        if row.get('completed') is False or not begin<=played<end or min(a,h)<0 or not all(math.isfinite(v) for v in (a,h)):continue
        identity=(row.get('league'),row.get('id'),row.get('away'),row.get('home'),played.isoformat())
        if identity in seen:continue
        seen.add(identity)
        if row.get('neutral') is True:continue
        targets=[(row.get('home'),'home'),(row.get('away'),'away')]
        if not isinstance(row.get('neutral'),bool):
            for tid,venue in targets:records[tid][venue]['unknown_venues']+=1
            continue
        try: handicap=float(row['home_handicap'])
        except (ValueError,TypeError,KeyError):handicap=None
        if handicap is None or not math.isfinite(handicap):
            for tid,venue in targets:records[tid][venue]['missing_lines']+=1
            continue
        home_cover=h-a+handicap
        for tid,venue,cover in [(row.get('home'),'home',home_cover),(row.get('away'),'away',-home_cover)]:
            records[tid][venue]['pushes' if abs(cover)<1e-8 else 'wins' if cover>0 else 'losses']+=1
    involved={row.get('league') for row in rows}
    for tid,team in teams.items():
        if involved and team.get('league') not in involved:continue
        values=records[tid]
        for record in values.values():
            n=record['wins']+record['losses'];record['graded_games']=n+record['pushes']
            record['cover_rate']=record['wins']/n if n else None
            record['record']=f"{record['wins']}-{record['losses']}-{record['pushes']}" if record['graded_games'] else None
        team['ats_records']=dict(**values,window_start=begin.isoformat(),as_of=end.isoformat(),source=source,missing_years=list(missing_years))


def comparison(data, game, params):
    par={**DEFAULTS,**params[game['league']]}
    result=dict(active=False,margin_adjustment=0.,status='ATS history unavailable; refresh the slate to load it.')
    if game.get('neutral'):
        result['status']='Neutral venue: home/away ATS adjustment skipped.';return result
    if par['ats_weight']==0 or par['ats_cap_points']==0:
        result['status']='ATS history adjustment disabled by its weight or cap.';return result
    try: cutoff=timestamp(game.get('kickoff'))
    except ValueError:return result
    rates={}
    for side,venue in [('away','away'),('home','home')]:
        history=data['teams'].get(game[side],{}).get('ats_records',{})
        result[side]=history
        record=history.get(venue,{})
        try:
            if timestamp(history.get('as_of'))>cutoff:raise ValueError('Future ATS data')
            wins=float(record['wins']);losses=float(record['losses']);missing=float(record.get('missing_lines',0))+float(record.get('unknown_venues',0));n=wins+losses
            if min(wins,losses,missing)<0 or not all(math.isfinite(v) for v in (wins,losses,missing)) or n<5 or history.get('missing_years') or n/(n+missing)<.75:raise ValueError('Insufficient ATS coverage')
        except (ValueError,TypeError,KeyError):
            result['status']='ATS adjustment needs at least 5 non-push games per relevant venue, 75% known-line coverage, all requested years and pregame timestamps.';return result
        rates[side]=(wins+par['ats_prior_games']/2)/(n+par['ats_prior_games'])
    delta=par['ats_weight']*(rates['away']-rates['home'])
    result.update(active=True,smoothed_cover_rates=rates,margin_adjustment=max(-par['ats_cap_points'],min(par['ats_cap_points'],delta)),status='Venue-specific past ATS rates, smoothed toward 50%; point weight is provisional.')
    return result
