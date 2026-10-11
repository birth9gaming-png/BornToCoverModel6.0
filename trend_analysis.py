"""Fixed football trend screens, with later games kept separate from selection."""
from __future__ import annotations
from collections import defaultdict
from statistics import median,NormalDist
from datetime import datetime,timezone,timedelta
import math
from audit_time import observed_time

PATTERNS=('All games','Home games','Road games','Favored by 3 or more','Home favorite by 3 or more',
    'Road favorite by 3 or more','Underdog by 7 or more','After a win','After a loss',
    'Home after a road loss','After a ranked-opponent win','Ranked opponent next',
    'Prior offense averages 30+ points','Prior offense success rate 45%+')


def records(data,league,historical=None):
    result={str(r['id']):dict(r) for r in data.get('trend_history',[]) if r.get('league')==league}
    if historical and league=='CFB':
        quotes=defaultdict(list)
        for q in historical.get('lines',[]):quotes[q['id']].append(q)
        for g in historical.get('games',[]):
            lines=[q for q in quotes[g['id']] if q['away']==g['away'] and q['home']==g['home']]
            spreads=[q['home_handicap'] for q in lines if q.get('home_handicap') is not None]
            totals=[q['total'] for q in lines if q.get('total') is not None]
            result[str(g['id'])]=dict(g,league='CFB',away='CFB:'+g['away'],home='CFB:'+g['home'],
                home_handicap=median(spreads) if spreads else None,market_total=median(totals) if totals else None)
    elif historical and league=='NFL':
        for g in historical.get('games',[]):result[str(g['id'])]=dict(g,league='NFL',away='NFL:'+g['away'],home='NFL:'+g['home'],home_handicap=-g['line_away'] if g.get('line_away') is not None else None)
    return list(result.values())


def _poll(polls,season,week,team,at):
    if week is None:return None
    eligible=[p for p in polls if p.get('season')==season and p.get('week',999)<week]
    checked=[]
    for p in eligible:
        if p.get('published_at'):
            try:
                if observed_time(p['published_at'])>=at:continue
            except ValueError:continue
        checked.append(p)
    p=max(checked,key=lambda p:p['week'],default=None)
    return (team in p['ranks']) if p else None


def _rates(rows,key,reverse=False):
    values=[r[key] for r in rows if r[key] is not None]
    wins=sum(v<0 if reverse else v>0 for v in values);losses=sum(v>0 if reverse else v<0 for v in values);pushes=sum(v==0 for v in values)
    n=wins+losses
    return dict(wins=wins,losses=losses,pushes=pushes,record=f'{wins}-{losses}-{pushes}',decisions=n,rate=wins/n if n else None)


def _bound(rate,n,comparisons):
    if not n:return None
    # A family-wise Wilson interval acknowledges that several patterns are screened.
    z=NormalDist().inv_cdf(1-.05/(2*comparisons));den=1+z*z/n
    return (rate+z*z/(2*n)-z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n)))/den


def scan(data,league,team,historical=None,as_of=None,min_decisions=20,start_season=2016,assumed_odds=-110.):
    as_of=observed_time(as_of) if as_of else datetime.now(timezone.utc)
    if not math.isfinite(assumed_odds) or abs(assumed_odds)<100:raise ValueError('Use valid American odds.')
    profit=100/abs(assumed_odds) if assumed_odds<0 else assumed_odds/100
    break_even=1/(1+profit)
    all_rows=records(data,league,historical)
    def kickoff(r):
        try:return observed_time(r['kickoff'])
        except (ValueError,KeyError):return datetime.max.replace(tzinfo=timezone.utc)
    team_rows=sorted([g for g in all_rows if team in (g['away'],g['home'])],key=kickoff)
    polls=(historical or {}).get('polls',[])+data.get('poll_history',[])
    statmap={(r['id'],'CFB:'+r['team']):r for r in (historical or {}).get('advanced',[])} if league=='CFB' else {(r['id'],'NFL:'+r['team']):r for r in (historical or {}).get('team_games',[])}
    buckets={k:[] for k in PATTERNS};last=[];eligible_count=0
    for index,g in enumerate(team_rows):
        at=kickoff(g);season=g.get('season',at.year)
        if at+timedelta(hours=12)>=as_of or season<start_season or g.get('completed') is False:continue
        a,h=g.get('away_score'),g.get('home_score')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (a,h)):continue
        previous=[p for p in last if p.get('season',kickoff(p).year)==season]
        recent=previous[-1] if previous else None
        opponent=g['home'] if g['away']==team else g['away'];home=g['home']==team;neutral=g.get('neutral')
        margin=a-h if not home else h-a
        handicap=g.get('home_handicap');handicap=handicap if home else -handicap if handicap is not None else None
        total=g.get('market_total')
        point=lambda p:p['home_score'] if p['home']==team else p['away_score']
        prev_margin=(point(recent)-(recent['away_score'] if recent['home']==team else recent['home_score'])) if recent else None
        within=bool(recent and 1<=(at-kickoff(recent)).days<=10)
        next_games=[row for row in team_rows[index+1:] if row.get('season',kickoff(row).year)==season]
        following=next_games[0] if next_games else None
        next_opp=following['home'] if following and following['away']==team else following['away'] if following else None
        next_ranked=_poll(polls,season,g.get('week',999),next_opp,at) if next_opp else None
        current_ranked=_poll(polls,season,g.get('week',999),opponent,at)
        prev_opp=recent['home'] if recent and recent['away']==team else recent['away'] if recent else None
        prior_ranked=_poll(polls,season,recent.get('week',999),prev_opp,kickoff(recent)) if recent else None
        successes=plays=0.
        for past in previous:
            stats=statmap.get((past['id'],team))
            if not stats:continue
            if league=='CFB':plays+=stats['offense']['plays'];successes+=stats['offense']['plays']*stats['offense']['successRate']
            else:plays+=stats['plays'];successes+=stats['successes']
        tests={'All games':True,'Home games':home and neutral is False,'Road games':not home and neutral is False,
            'Favored by 3 or more':handicap is not None and handicap<=-3,
            'Home favorite by 3 or more':home and neutral is False and handicap is not None and handicap<=-3,
            'Road favorite by 3 or more':not home and neutral is False and handicap is not None and handicap<=-3,
            'Underdog by 7 or more':handicap is not None and handicap>=7,
            'After a win':within and prev_margin>0,'After a loss':within and prev_margin<0,
            'Home after a road loss':home and neutral is False and within and prev_margin<0 and recent['away']==team and recent.get('neutral') is False,
            'After a ranked-opponent win':within and prev_margin>0 and prior_ranked is True,
            'Ranked opponent next':bool(following and 1<=(kickoff(following)-at).days<=10 and next_ranked is True and current_ranked is False),
            'Prior offense averages 30+ points':len(previous)>=2 and sum(point(p) for p in previous)/len(previous)>=30,
            'Prior offense success rate 45%+':len(previous)>=2 and plays>0 and successes/plays>=.45}
        entry=dict(id=g['id'],kickoff=g['kickoff'],su=margin,ats=margin+handicap if handicap is not None else None,
                   total=a+h-total if total is not None else None)
        eligible_count+=1
        for pattern,qualifies in tests.items():
            if qualifies:buckets[pattern].append(entry)
        last.append(g)
    output=[];comparisons=len(PATTERNS)*2
    for pattern,rows in buckets.items():
        split=int(len(rows)*.7);older,later=rows[:split],rows[split:]
        su=_rates(rows,'su');ats=_rates(rows,'ats');over=_rates(rows,'total');under=_rates(rows,'total',True)
        for market,key in [('ATS','ats'),('Total','total')]:
            train_base=_rates(older,key);reverse=train_base['rate'] is not None and train_base['rate']<.5
            train,test=_rates(older,key,reverse),_rates(later,key,reverse)
            direction=('Fade team' if reverse else 'Back team') if market=='ATS' else ('Under' if reverse else 'Over')
            lower=_bound(test['rate'],test['decisions'],comparisons) if test['rate'] is not None else None
            adequate=train['decisions']>=min_decisions and test['decisions']>=max(10,min_decisions//2)
            supported=adequate and train['rate']>break_even and lower>break_even
            roi=(profit*test['wins']-test['losses'])/max(1,test['wins']+test['losses']+test['pushes'])
            output.append(dict(pattern=pattern,market=market,selected_direction=direction,games=len(rows),
                ATS_WLP=ats['record'],over_WLP=over['record'],under_WLP=under['record'],outright_WLT=su['record'],
                missing_lines=len(rows)-(ats['wins']+ats['losses']+ats['pushes']) if market=='ATS' else len(rows)-(over['wins']+over['losses']+over['pushes']),
                older_record=train['record'],later_record=test['record'],later_win_rate=test['rate'],
                later_ROI_at_assumed_price=roi if test['decisions'] else None,assumed_odds=assumed_odds,
                break_even_nonpush=break_even,later_lower_bound=lower,
                status='Later-game support' if supported else 'Small sample' if not adequate else 'Needs more evidence',
                split_date=later[0]['kickoff'] if later else None))
    return output,dict(team=team,eligible_games=eligible_count,patterns_screened=comparisons,as_of=as_of.isoformat(),price_assumption=assumed_odds)
