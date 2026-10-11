"""Timestamped quote logging and same-book closing-snapshot comparisons."""
from __future__ import annotations
import copy
import csv
import hashlib
import io
import json
import math
from datetime import datetime,timezone,timedelta
from matchup_history import timestamp
from audit_time import observed_time


def capture(data,results,league='CFB',now=None):
    now=now or datetime.now(timezone.utc);output=[]
    for r in results:
        if r['league']!=league or not r.get('markets') or 'away' not in r:continue
        g=r['game']
        try:kickoff=observed_time(g.get('kickoff'))
        except ValueError:continue
        if now>=kickoff:continue
        q=max(r['markets'],key=lambda v:v.get('selection_score',v['conditional_ev']) if league=='NFL' else v['conditional_ev'])
        if q['conditional_ev']<=0:continue
        try:quote_time=observed_time(q.get('updated_at'))
        except ValueError:continue
        if quote_time>now or now-quote_time>timedelta(hours=6):continue
        if not q.get('book') or q['book']=='Manual scenario':continue
        payload=f"{league}|{g['id']}|{q['book']}|{q['kind']}|{q['selection']}|{now.isoformat()}"
        output.append(dict(log_id=hashlib.sha256(payload.encode()).hexdigest()[:20],league=league,game_id=str(g['id']),
                           matchup=g['away']+' @ '+g['home'],kickoff=kickoff.isoformat(),captured_at=now.isoformat(),quote_as_of=quote_time.isoformat(),
                           book=q['book'],market=q['kind'],side=q['selection'],line=q['line'],odds=q['odds'],
                           projected_margin=r['margin_away'],projected_total=r['fair_total'],p_win=q['p_win'],p_push=q['p_push'],conditional_ev=q['conditional_ev'],
                           model_version='Prototype 10',fitted_model_hash=hashlib.sha256(json.dumps(data.get(league.lower()+'_boost_model'),sort_keys=True,allow_nan=False).encode()).hexdigest(),
                           auto_injury_settings=copy.deepcopy(data.get('auto_injury_settings',{})),
                           **({'nfl_layer_settings':copy.deepcopy(data.get('nfl_layer_settings',{}))} if league=='NFL' else {}),
                           parameters=copy.deepcopy(data['parameters'])))
    return sorted(output,key=lambda v:v['conditional_ev'],reverse=True)[:2]


def read_closes(filename,content):
    text=content.decode('utf-8-sig')
    if filename.lower().endswith('.csv'):rows=list(csv.DictReader(io.StringIO(text)))
    else:
        value=json.loads(text);rows=value.get('closing_quotes',[]) if isinstance(value,dict) else value
    if not isinstance(rows,list):raise ValueError('Closing quotes must be a list.')
    output=[]
    for row in rows:
        quote=dict(game_id=str(row['game_id']),book=str(row['book']),market=str(row['market']).lower(),side=str(row['side']).lower(),
                   line=float(row['line']),odds=float(row['odds']),quoted_at=observed_time(row['quoted_at']).isoformat())
        if quote['market'] not in ('spread','total') or quote['side'] not in (('away','home') if quote['market']=='spread' else ('over','under')):
            raise ValueError('Invalid market or side in closing quote.')
        if not math.isfinite(quote['line']) or not math.isfinite(quote['odds']) or abs(quote['odds'])<100:raise ValueError('Invalid line or American odds.')
        if quote['market']=='total' and quote['line']<=0:raise ValueError('Total must be positive.')
        output.append(quote)
    return output


def compare(taken,close,now=None):
    now=now or datetime.now(timezone.utc)
    for key in ('game_id','book','market','side'):
        if str(taken[key])!=str(close[key]):raise ValueError('Closing snapshot must match the same game, book, market and side.')
    kickoff=observed_time(taken['kickoff']);quoted=observed_time(close['quoted_at'])
    if not observed_time(taken['captured_at'])<=quoted<=min(now,kickoff):raise ValueError('Closing snapshot must be after capture, before kickoff and not in the future.')
    if kickoff-quoted>timedelta(minutes=60):raise ValueError('Closing snapshot must be within 60 minutes before kickoff.')
    a,b=float(taken['line']),float(close['line'])
    if not all(math.isfinite(v) for v in (a,b,float(taken['odds']),float(close['odds']))):raise ValueError('Lines and odds must be finite.')
    point_clv=a-b if taken['market']=='spread' or taken['side']=='under' else b-a
    output=dict(points=point_clv,minutes_before_kickoff=(kickoff-quoted).total_seconds()/60,
                closing_line=b,closing_odds=float(close['odds']),closing_quoted_at=quoted.isoformat(),same_line=math.isclose(a,b))
    def implied(odds):return abs(odds)/(abs(odds)+100) if odds<0 else 100/(odds+100)
    output['price_clv_implied_probability']=implied(float(close['odds']))-implied(float(taken['odds'])) if output['same_line'] else None
    return output


def summaries(ledger,closes,now=None):
    output=[]
    for item in ledger:
        candidates=[];reason='No matching closing snapshot.'
        for close in closes:
            if any(str(item[k])!=str(close.get(k)) for k in ('game_id','book','market','side')):continue
            try:candidates.append(compare(item,close,now))
            except ValueError as exc:reason=str(exc)
        valid=max(candidates,key=lambda c:timestamp(c['closing_quoted_at']),default=None)
        output.append(dict(log_id=item['log_id'],matchup=item['matchup'],book=item['book'],market=item['market'],side=item['side'],
                           taken_line=item['line'],captured_at=item['captured_at'],status='Compared closing snapshot' if valid else reason,
                           **(valid or {})))
    return output
