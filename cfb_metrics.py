"""CFB possession ratings, preseason transfer aggregates and manual roster inputs."""
from __future__ import annotations
import copy
import math
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from model import CFB_METRICS, ensure_cfb_defaults, snapshot_season, validate

INPUT_FIELDS = CFB_METRICS + ('fei_season', 'fei_through_week', 'roster_season', 'fei_source', 'roster_source')


def normalized(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


class RatingTable(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.cell=None; self.text=[]

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row=[]
        elif tag in ('td', 'th') and self.row is not None: self.cell=[]

    def handle_data(self, value):
        self.text.append(value)
        if self.cell is not None: self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split())); self.cell=None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row); self.row=None


def parse_fei(html, season):
    parser=RatingTable(); parser.feed(html)
    text=' '.join(parser.text)
    if not re.search(r'\b' + str(season) + r'\s+FEI\s+Ratings', text, re.I):
        raise ValueError('FEI page does not identify the requested season.')
    found=re.search(r'through\s+Week\s+(\d+)', text, re.I)
    if found is None:
        raise ValueError('FEI page has no through-week label; use a dated manual import.')
    week=int(found.group(1)); columns=None; ratings={}
    for row in parser.rows:
        if all(k in row for k in ('Team', 'OFEI', 'DFEI')):
            columns={k:row.index(k) for k in ('Team', 'OFEI', 'DFEI')}; continue
        if columns is None or len(row) <= max(columns.values()) or not row[0].isdigit(): continue
        try: off, defense = float(row[columns['OFEI']]), float(row[columns['DFEI']])
        except ValueError: continue
        if not all(math.isfinite(v) for v in (off, defense)): continue
        name=row[columns['Team']]
        if normalized(name) in ratings: raise ValueError('Duplicate team in FEI table.')
        ratings[normalized(name)]=dict(ofei=off,dfei=defense,fei_season=season,fei_through_week=week)
    if not ratings: raise ValueError('FEI table could not be parsed; ratings were not guessed.')
    return ratings, week


def attach_fei(teams, html, season, cutoff=None):
    ratings, week=parse_fei(html, season)
    if cutoff is not None and week > cutoff:
        raise ValueError(f'Latest FEI is through week {week}, after cutoff {cutoff}. Import an archived pregame rating instead.')
    matched=0
    for team in teams.values():
        if team.get('league') != 'CFB': continue
        values=ratings.get(normalized(team.get('name', '')))
        if values:
            team.update(values,fei_source=f'https://bcftoys.com/{season}-fei',fei_loaded_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            matched+=1
    if not matched: raise ValueError('No FEI team names matched the CFB teams in this snapshot.')
    return matched, week


def portal_ratings(rows, season, preseason_cutoff):
    """Sum provider ratings in minus out; only dated, confirmed immediate transfers."""
    totals={}; seen=set()
    def start(name):
        key=normalized(name)
        totals.setdefault(key, dict(portal_net_rating=0., portal_incoming=0, portal_outgoing=0,
                                    portal_incomplete_transfers=0, portal_missing_reasons={}))
        return key
    def incomplete(names, reasons):
        for name in names:
            values=totals[start(name)]
            values['portal_incomplete_transfers']+=1
            for reason in reasons:
                counts=values['portal_missing_reasons']
                counts[reason]=counts.get(reason,0)+1
    for row in rows:
        if row.get('season') != season: continue
        origin, destination=row.get('origin'),row.get('destination')
        if not origin or not destination or normalized(origin)==normalized(destination): continue
        if row.get('eligibility') == 'Withdrawn': continue
        date=row.get('transferDate')
        identity=(row.get('firstName'),row.get('lastName'),row.get('position'),normalized(origin),normalized(destination),date)
        if identity in seen: continue
        seen.add(identity)
        reasons=[]
        try:
            transfer=datetime.fromisoformat(date.replace('Z','+00:00'))
            if transfer.tzinfo is None: transfer=transfer.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, AttributeError):
            transfer=None; reasons.append('Missing or invalid transfer date')
        if transfer is not None and transfer >= preseason_cutoff: continue
        if row.get('eligibility') != 'Immediate':
            reasons.append('Eligibility '+str(row.get('eligibility') or 'unknown')+' (Immediate required)')
        try: rating=float(row.get('rating'))
        except (ValueError, TypeError): rating=None
        if rating is None or not math.isfinite(rating) or rating < 0:
            reasons.append('Missing or invalid player rating')
        if reasons:
            incomplete((origin,destination),reasons); continue
        a,h=start(origin),start(destination)
        totals[a]['portal_net_rating']-=rating; totals[a]['portal_outgoing']+=1
        totals[h]['portal_net_rating']+=rating; totals[h]['portal_incoming']+=1
    for values in totals.values():
        if values['portal_incomplete_transfers']:
            values['portal_net_rating']=None
    return totals


def attach_portal(teams, rows, season, preseason_cutoff):
    if not rows: raise ValueError('Portal feed returned no records; net ratings remain unknown.')
    totals=portal_ratings(rows,season,preseason_cutoff)
    incomplete=0
    for team in teams.values():
        if team.get('league') != 'CFB': continue
        values=totals.get(normalized(team.get('name','')),dict(portal_net_rating=0.,portal_incoming=0,portal_outgoing=0,portal_incomplete_transfers=0,portal_missing_reasons={}))
        team.update(values,roster_season=season,portal_source='CFBD /player/portal: confirmed immediate transfers, native rating sum in minus out',portal_cutoff=preseason_cutoff.isoformat())
        if values['portal_net_rating'] is None: incomplete+=1
    return incomplete


def roster_coverage(data):
    """Recompute coverage for slate teams after refreshes and manual overrides."""
    involved={g[side] for g in data.get('games',[]) if g.get('league')=='CFB' for side in ('away','home')}
    season=snapshot_season(data); rows=[]; missing_starters=0; missing_portal=0
    for tid in sorted(involved):
        team=data['teams'].get(tid,{})
        games=team.get('roster_games_played')
        if games is None: games=team.get('sample_games')
        context=[]
        if season is not None and team.get('roster_season')!=season: context.append('Roster season missing or mismatched')
        if games is None: context.append('Games played missing')
        missing=[label for key,label in [('returning_starters_off','offensive starters'),('returning_starters_def','defensive starters')] if team.get(key) is None]
        starter_status='Available' if not missing and not context else '; '.join(['Missing '+', '.join(missing)] if missing else [])
        if context: starter_status='; '.join(filter(None,[starter_status,*context]))
        if missing or context: missing_starters+=1
        portal_status='Available'
        if team.get('portal_net_rating') is None:
            reasons=team.get('portal_missing_reasons') or {}
            portal_status='; '.join(f'{reason}: {count} transfer(s)' for reason,count in sorted(reasons.items())) or 'Net rating missing; refresh to load transfer diagnostics'
        if context: portal_status+='; '+'; '.join(context)
        if team.get('portal_net_rating') is None or context: missing_portal+=1
        rows.append(dict(team_id=tid,team=team.get('name',tid),returning_offense=team.get('returning_starters_off'),returning_defense=team.get('returning_starters_def'),starter_status=starter_status,portal_net_rating=team.get('portal_net_rating'),portal_status=portal_status,games_played=games))
    return dict(slate_teams=len(involved),missing_starters=missing_starters,missing_portal=missing_portal,rows=rows)


def current_feed_warnings(data):
    """Replace old static roster warnings with live coverage in the dashboard."""
    return [w for w in data.get('warnings',[]) if not (
        w.startswith('Returning-starter counts require your verified inputs') or
        re.match(r'^\d+ CFB teams have incomplete portal ratings, dates or eligibility;',w))]


def apply_cfb_inputs(data, records, remember=True, ignore_nulls=False):
    """Join exact normalized names/IDs; bulk templates may leave nulls untouched."""
    result=copy.deepcopy(data); ensure_cfb_defaults(result)
    if not isinstance(records,dict): raise ValueError('CFB input file needs a teams object.')
    names={}
    for tid, team in result['teams'].items():
        if team.get('league') != 'CFB': continue
        for name in (tid,team.get('name',tid)):
            names.setdefault(normalized(name),set()).add(tid)
    unmatched=[]; updates=[]; targeted=set()
    for name,record in records.items():
        if not isinstance(record,dict): raise ValueError('Each team input must be an object.')
        matches=names.get(normalized(name),set())
        if len(matches)!=1: unmatched.append(name); continue
        tid=next(iter(matches))
        if tid in targeted: raise ValueError('More than one input record matches ' + tid)
        targeted.add(tid)
        values={k:v for k,v in record.items() if k in INPUT_FIELDS and (v is not None or not ignore_nulls)}
        # Forms can explicitly clear a metric. Blank bulk-template fields preserve it.
        result['teams'][tid].update(values)
        if remember:
            result.setdefault('cfb_overrides',{}).setdefault(result['teams'][tid].get('name',tid),{}).update(values)
        updates.append(tid)
    validate(result)
    if unmatched: result.setdefault('warnings',[]).append('CFB inputs not joined: ' + ', '.join(unmatched))
    return result, updates, unmatched


def preserve_cfb_overrides(previous, next_data):
    season=snapshot_season(next_data); accepted={}
    for name, record in previous.get('cfb_overrides',{}).items():
        values=copy.deepcopy(record)
        if values.get('roster_season') != season:
            values={k:v for k,v in values.items() if k.startswith('fei_') or k in ('ofei','dfei')}
        if values.get('fei_season') != season:
            values={k:v for k,v in values.items() if not k.startswith('fei_') and k not in ('ofei','dfei')}
        # Games played advances with the newly loaded statistics, unless edited again.
        values.pop('roster_games_played',None)
        if values: accepted[name]=values
    if accepted:
        result,_,_=apply_cfb_inputs(next_data,accepted)
        return result
    return next_data
