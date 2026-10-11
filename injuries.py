"""Reviewable injury imports. Player value is an explicit user assumption."""
from __future__ import annotations
import csv
import io
import json
import math
import re
from datetime import datetime,timezone,timedelta
from data_pipeline import normal_name
from matchup_history import timestamp
from audit_time import observed_time

STATUSES=('Out','Doubtful','Questionable','Probable','Available')
STATUS_ALIASES={'out':'Out','doubtful':'Doubtful','questionable':'Questionable','probable':'Probable',
                'available':'Available','active':'Available','healthy':'Available'}


def parse_text(text,team_names,default_team=None):
    """Read lines containing an explicit status; uncertain lines stay unparsed."""
    current=default_team;rows=[];ignored=0
    known={normal_name(name):name for name in team_names}
    for line in text.splitlines():
        line=line.strip()
        if normal_name(line) in known:current=known[normal_name(line)];continue
        match=re.search(r'\b(Out|Doubtful|Questionable|Probable|Available)\s*$',line,re.I)
        if not match:continue
        player=line[:match.start()].strip(' \t|:–-')
        if not current or not player or len(player)>120:ignored+=1;continue
        rows.append(dict(team=current,player=player,status=STATUS_ALIASES[match.group().strip().lower()],report_as_of='',point_impact=0.))
    return rows,ignored


def read_report(filename,content,team_names,default_team=None):
    if len(content)>10*1024*1024:raise ValueError('Injury file exceeds 10 MB.')
    if filename.lower().endswith('.pdf'):
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(content))
        if len(reader.pages)>50:raise ValueError('Injury PDF has more than 50 pages.')
        text='\n'.join(page.extract_text(extraction_mode='layout') or '' for page in reader.pages)
        rows,ignored=parse_text(text,team_names,default_team)
        if not rows:raise ValueError('No readable player/status rows found. Use a text-based PDF or the CSV template; scanned PDFs need OCR first.')
    else:
        text=content.decode('utf-8-sig');ignored=0
        if filename.lower().endswith('.json'):
            payload=json.loads(text);rows=payload.get('injuries',[]) if isinstance(payload,dict) else payload
        elif filename.lower().endswith('.csv'):rows=list(csv.DictReader(io.StringIO(text)))
        else:rows,ignored=parse_text(text,team_names,default_team)
        if not isinstance(rows,list):raise ValueError('Expected an injury JSON array or an injuries list.')
    output=[]
    for row in rows:
        if not isinstance(row,dict):ignored+=1;continue
        output.append(dict(team=row.get('team') or default_team or '',player=row.get('player',''),
                           status=row.get('status',''),report_as_of=row.get('report_as_of',''),
                           point_impact=row.get('point_impact',0.),source=filename))
    return output,ignored


def review_rows(rows,data,league):
    known={}
    for tid,team in data['teams'].items():
        if team.get('league')!=league:continue
        for name in (tid,team.get('name',tid)):known.setdefault(normal_name(name),set()).add(tid)
    approved=[];errors=[]
    for i,row in enumerate(rows):
        try:
            matches=known.get(normal_name(str(row.get('team',''))),set())
            if len(matches)!=1:raise ValueError('team must uniquely match a loaded '+league+' team')
            player=str(row.get('player','')).strip()
            if not player or player.lower()=='nan':raise ValueError('player name is required')
            status=STATUS_ALIASES.get(str(row.get('status','')).strip().lower())
            if status is None:raise ValueError('status must be Out, Doubtful, Questionable, Probable or Available')
            published=observed_time(row.get('report_as_of'))
            if published>datetime.now(timezone.utc):raise ValueError('report publication cannot be in the future')
            impact=float(row.get('point_impact') or 0.)
            if not math.isfinite(impact) or not -20<=impact<=0:raise ValueError('own-team score impact must be between −20 and 0')
            approved.append(dict(league=league,team_id=next(iter(matches)),team=str(row['team']),player=player,
                                 status=status,report_as_of=published.isoformat(),point_impact=0. if status=='Available' else impact,
                                 source=str(row.get('source','Reviewed import')),reviewed=True))
        except (ValueError,TypeError) as exc:errors.append(f'Row {i+1}: {exc}')
    return approved,errors


def effects(rows,game,max_age_hours=96,cap=10.,now=None):
    now=now or datetime.now(timezone.utc);kickoff=observed_time(game.get('kickoff'));latest={};ignored=0
    for row in rows:
        try:published=timestamp(row.get('report_as_of'))
        except ValueError:ignored+=1;continue
        if not row.get('reviewed') or row.get('league')!=game['league'] or row.get('team_id') not in (game['away'],game['home']):continue
        # An old report does not certify a healthy team or a future absence.
        if published>min(now,kickoff) or kickoff-published>timedelta(hours=max_age_hours):ignored+=1;continue
        key=(row['team_id'],normal_name(row['player']))
        if key not in latest or published>timestamp(latest[key]['report_as_of']):latest[key]=row
        elif published==timestamp(latest[key]['report_as_of']) and row!=latest[key]:
            raise ValueError('Conflicting same-time injury reports for '+row['player']+'. Remove or correct the conflicting row.')
    result=dict(away=0.,home=0.,rows=list(latest.values()),ignored=ignored,coverage={})
    for side in ('away','home'):
        eligible=[r for r in latest.values() if r['team_id']==game[side]]
        result[side]=max(-cap,sum(float(r['point_impact']) for r in eligible))
        result['coverage'][side]=f'{len(eligible)} reviewed player rows' if eligible else 'Unknown: no recent reviewed report rows'
    return result


def apply_to_games(data,league,now=None):
    """Compute all updates first, preserving manually edited injury scenarios."""
    updates=[]
    for game in data['games']:
        if game.get('league')!=league:continue
        try:observed_time(game.get('kickoff'))
        except ValueError:
            report=dict(rows=[],away=0.,home=0.,ignored=0,coverage={'away':'Unknown: verified kickoff missing; refresh schedule','home':'Unknown: verified kickoff missing; refresh schedule'})
        else:report=effects(data.get('injury_reports',[]),game,now=now)
        changed={};old=game.get('injury_import',{});report['applied_sides']={}
        for side in ('away','home'):
            applies=any(row['team_id']==game[side] for row in report['rows'])
            report['applied_sides'][side]=applies
            if applies:changed['injury_'+side]=report[side]
            elif old.get('applied_sides',{}).get(side) and game.get('injury_'+side)==old.get(side):changed['injury_'+side]=0.
        changed['injury_import']=report;updates.append((game,changed))
    for game,changed in updates:game.update(changed)
