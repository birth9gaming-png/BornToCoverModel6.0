"""AP snapshots with conservative prior-week cutoffs when publication time is absent."""
from __future__ import annotations
import re
from audit_time import observed_time


def snapshots(rows):
    result=[]
    for row in rows:
        season=int(row.get('season',row.get('year',0)));week=int(row.get('week',-1))
        if str(row.get('seasonType','regular')).lower()!='regular' or season<1900 or week<0:continue
        for poll in row.get('polls',[]):
            name=re.sub('[^a-z0-9]','',str(poll.get('poll','')).lower())
            if name not in ('aptop25','appoll','associatedpress'):continue
            ranks={}
            for entry in poll.get('ranks',[]):
                team=entry.get('school');rank=entry.get('rank')
                if team and isinstance(rank,(int,float)) and 1<=rank<=25:ranks['CFB:'+str(team)]=int(rank)
            if ranks:result.append(dict(id=f'AP-{season}-{week}',season=season,week=week,poll='AP Top 25',ranks=ranks,published_at=row.get('published_at')))
    return result


def ranking(data,team,game,through_week=None):
    cutoff=game.get('feature_through_week') if through_week is None else through_week
    season=data.get('season');eligible=[]
    for row in data.get('poll_history',[]):
        if row.get('season')!=season or cutoff is None or row.get('week',999)>cutoff:continue
        if row.get('published_at'):
            try:
                if observed_time(row['published_at'])>=observed_time(game['kickoff']):continue
            except (ValueError,KeyError):continue
        eligible.append(row)
    selected=max(eligible,key=lambda r:r['week'],default=None)
    if selected is None:return None,None,'No eligible AP snapshot; ranking unknown'
    rank=selected['ranks'].get(team)
    return rank,rank is not None,f"AP Top 25 week {selected['week']}; conservative prior-week snapshot"
