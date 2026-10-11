"""Checked, local-only imports of CFBD game-level exports."""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from data_pipeline import normal_name, verified_cfb_home_handicap
from matchup_history import timestamp
from audit_time import observed_time

FORMAT = 'cfbd-history-v1'
MAX_BYTES = 200 * 1024 * 1024


def nested(value):
    if isinstance(value, str) and value.strip().startswith(('[', '{')):
        try: return json.loads(value)
        except (ValueError, TypeError):
            try: return ast.literal_eval(value)
            except (ValueError, SyntaxError): pass
    return value


def flat(row, prefix=''):
    result = {}
    for key, value in row.items():
        name = prefix + str(key)
        value = nested(value)
        if isinstance(value, dict): result.update(flat(value, name + '.'))
        else: result[normal_name(name)] = value
    return result


def pick(row, *names):
    for name in names:
        value = row.get(normal_name(name))
        if value is not None and not (isinstance(value, str) and value.strip().lower() in ('', 'null', 'none', 'nan')):
            return value
    return None


def numeric(value):
    try: n = float(value)
    except (ValueError, TypeError): return None
    return n if math.isfinite(n) else None


def integer(value):
    n = numeric(value)
    return int(n) if n is not None and n == int(n) else None


def boolean(value):
    if isinstance(value, bool): return value
    if str(value).strip().lower() in ('true', '1'): return True
    if str(value).strip().lower() in ('false', '0'): return False
    return None


def game_id(row):
    value = pick(row, 'gameId', 'id')
    n = integer(value)
    return str(n) if n is not None else str(value).strip() if value is not None else None


def team_name(value):
    return str(value).strip() if value is not None and str(value).strip() else None


def normalize_game(row):
    r = flat(row)
    gid = game_id(r)
    season = integer(pick(r, 'season', 'year'))
    week = integer(pick(r, 'week'))
    away = team_name(pick(r, 'awayTeam', 'away'))
    home = team_name(pick(r, 'homeTeam', 'home'))
    kickoff = pick(r, 'startDate', 'kickoff')
    if not gid or season is None or week is None or not away or not home or away == home:
        raise ValueError('Games needs id, season, week, awayTeam and homeTeam.')
    try: played = observed_time(kickoff)
    except (ValueError, TypeError): raise ValueError('Games needs a valid startDate/kickoff with timezone.') from None
    a = numeric(pick(r, 'awayPoints', 'awayScore', 'away_score'))
    h = numeric(pick(r, 'homePoints', 'homeScore', 'home_score'))
    if any(v is not None and (v < 0 or v != int(v)) for v in (a, h)):
        raise ValueError('Final scores must be nonnegative integers.')
    finished = boolean(pick(r, 'completed'))
    if finished is None:
        finished = a is not None and h is not None and played < datetime.now(timezone.utc)
    season_type = str(pick(r, 'seasonType') or '').lower()
    return dict(id=gid, season=season, week=week, away=away, home=home,
                kickoff=played.isoformat(), away_score=a, home_score=h,
                completed=finished, neutral=boolean(pick(r, 'neutralSite', 'neutral')),
                season_type=season_type,
                away_classification=pick(r, 'awayClassification'),
                home_classification=pick(r, 'homeClassification'))


def normalize_advanced(row):
    r = flat(row); gid = game_id(r); team = team_name(pick(r, 'team', 'school'))
    if not gid or not team: raise ValueError('Advanced GAME stats needs gameId and team; season aggregates cannot be used here.')
    result = dict(id=gid, team=team, season=integer(pick(r, 'season', 'year')), week=integer(pick(r, 'week')))
    for side in ('offense', 'defense'):
        plays = numeric(pick(r, side + '.plays'))
        success = numeric(pick(r, side + '.successRate'))
        explosion = numeric(pick(r, side + '.explosiveness'))
        if plays is None or plays <= 0 or success is None or not 0 <= success <= 1 or explosion is None or explosion < 0:
            raise ValueError('Advanced GAME stats needs offense/defense plays, successRate (0–1) and explosiveness. Nested columns must contain JSON or flattened field names.')
        result[side] = dict(plays=plays, successRate=success, explosiveness=explosion)
    return result


def box_fragments(row):
    """Nested API teams, one-team flat rows, or category/stat rows."""
    r = flat(row); gid = game_id(r)
    entries = nested(pick(r, 'teams'))
    if isinstance(entries, list):
        for entry in entries:
            yield from box_fragments(dict(id=gid, **entry))
        return
    team = team_name(pick(r, 'team', 'school'))
    if not gid or not team: raise ValueError('Team GAME stats needs id and teams JSON, or id and a team column.')
    canonical = {key: numeric(pick(r, key)) for key in ('yards', 'plays', 'giveaways')}
    if all(v is not None for v in canonical.values()):
        yield gid, team, {'canonical': canonical}; return
    stats = nested(pick(r, 'stats'))
    values = {}
    if isinstance(stats, list):
        values = {normal_name(item.get('category', '')): item.get('stat') for item in stats if isinstance(item, dict)}
    elif isinstance(stats, dict): values = {normal_name(k): v for k, v in stats.items()}
    category = pick(r, 'category', 'statCategory')
    if category is not None: values[normal_name(category)] = pick(r, 'stat', 'statValue')
    for key in ('passingAttempts', 'completionAttempts', 'rushingAttempts', 'totalYards', 'netPassingYards', 'rushingYards', 'turnovers'):
        value = pick(r, key, 'stats.' + key)
        if value is not None: values[normal_name(key)] = value
    if not values: raise ValueError('Team GAME stats has no readable box-score categories. Export nested teams/stats as JSON or flattened columns.')
    yield gid, team, values


def normalize_box(gid, team, values):
    if 'canonical' in values: core = values['canonical']
    else:
        attempts = numeric(values.get('passingattempts'))
        if attempts is None:
            text = str(values.get('completionattempts', ''))
            if re.fullmatch(r'\d+\s*[-/]\s*\d+', text):
                attempts = numeric(re.split(r'[-/]', text)[-1].strip())
        rushes = numeric(values.get('rushingattempts'))
        yards = numeric(values.get('totalyards'))
        if yards is None:
            passing, rushing = numeric(values.get('netpassingyards')), numeric(values.get('rushingyards'))
            if passing is not None and rushing is not None: yards = passing + rushing
        core = dict(yards=yards, plays=attempts + rushes if attempts is not None and rushes is not None else None,
                    giveaways=numeric(values.get('turnovers')))
    if any(v is None for v in core.values()) or core['plays'] <= 0 or core['giveaways'] < 0:
        raise ValueError('Team GAME stats needs total yards, passing/rushing attempts and turnovers for each team.')
    return dict(id=gid, team=team, **core)


def line_rows(row):
    r = flat(row); gid = game_id(r)
    away, home = team_name(pick(r, 'awayTeam', 'away')), team_name(pick(r, 'homeTeam', 'home'))
    if not gid or not away or not home: raise ValueError('Betting Lines needs id/gameId, awayTeam and homeTeam.')
    quotes = nested(pick(r, 'lines'))
    if not isinstance(quotes, list): quotes = [row]
    for quote in quotes:
        q = flat(quote)
        if pick(q, 'lineSignVerified') is True:
            handicap = numeric(pick(q, 'homeHandicap'))
        else:
            raw = dict(spread=pick(q, 'spread'), formattedSpread=pick(q, 'formattedSpread'))
            aliases = {normal_name(away): away, normal_name(home): home}
            handicap = verified_cfb_home_handicap(raw, away, home, aliases)
        total = numeric(pick(q, 'overUnder', 'total'))
        if handicap is None and (total is None or total <= 0):
            raise ValueError('Betting Lines needs a verifiable formattedSpread or positive overUnder. No handicap sign is guessed.')
        yield dict(id=gid, away=away, home=home, provider=str(pick(q, 'provider') or 'Unspecified provider'),
                   home_handicap=handicap, total=total if total is not None and total > 0 else None,
                   line_sign_verified=handicap is not None)


def documents(files):
    total = 0
    for filename, content in files:
        if str(filename).lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = [m for m in archive.infolist() if not m.is_dir() and not m.filename.startswith('__MACOSX/')]
                if len(members) > 2000: raise ValueError('ZIP contains too many files; split the upload.')
                for member in members:
                    if not member.filename.lower().endswith(('.csv', '.json')): continue
                    total += member.file_size
                    if total > MAX_BYTES: raise ValueError('Upload exceeds 200 MB of uncompressed data; split the seasons.')
                    yield filename + '/' + member.filename, archive.read(member)
        else:
            total += len(content)
            if total > MAX_BYTES: raise ValueError('Upload exceeds 200 MB; split the seasons.')
            yield filename, content


def rows(content, filename):
    text = content.decode('utf-8-sig')
    if filename.lower().endswith('.csv'):
        return list(csv.DictReader(io.StringIO(text)))
    payload = json.loads(text)
    if isinstance(payload, list): return payload
    if isinstance(payload, dict) and payload.get('format') == FORMAT:
        return ([dict(_kind='games', **r) for r in payload.get('games', [])] +
                [dict(_kind='advanced', **r) for r in payload.get('advanced', [])] +
                [dict(_kind='box', **r) for r in payload.get('box', [])] +
                [dict(_kind='lines', **r) for r in payload.get('lines', [])] +
                [dict(_kind='polls', **r) for r in payload.get('polls', [])])
    raise ValueError('JSON must be a CFBD response array or an exported historical dataset.')


def kind(row):
    r = flat(row)
    explicit = pick(r, '_kind')
    if explicit: return explicit
    if row.get('polls') is not None:return 'polls'
    if pick(r, 'lines', 'formattedSpread', 'spread', 'overUnder') is not None: return 'lines'
    if pick(r, 'teams', 'stats', 'category', 'rushingAttempts', 'completionAttempts') is not None: return 'box'
    if pick(r, 'offense.plays', 'offense.successRate', 'offense') is not None: return 'advanced'
    if pick(r, 'homeTeam', 'home') is not None and pick(r, 'startDate', 'kickoff') is not None: return 'games'
    return 'unknown'


def import_files(files, existing=None):
    maps = {k: {} for k in ('games', 'advanced', 'box', 'lines','polls')}
    conflicts = {k: set() for k in maps}; fragments = {}; file_reports = []; issues = Counter()
    supplied = list(files)
    if existing is not None:
        supplied.insert(0, ('existing_history.json', json.dumps(existing, allow_nan=False).encode()))

    def keep(category, key, value):
        if key in conflicts[category]: return
        previous = maps[category].get(key)
        if previous is not None and previous != value:
            conflicts[category].add(key); maps[category].pop(key, None)
            issues[category + ': conflicting duplicate records excluded'] += 1
        else: maps[category][key] = value

    for filename, content in documents(supplied):
        report = dict(file=filename, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), rows=0, recognized=0, rejected=0, types={})
        counts = Counter()
        try: records = rows(content, filename)
        except (ValueError, UnicodeError, csv.Error) as exc:
            report['rejected'] = 1; issues[str(exc)] += 1; file_reports.append(report); continue
        report['rows'] = len(records)
        for row in records:
            try:
                if not isinstance(row, dict): raise ValueError('Expected objects with named fields.')
                category = kind(row); counts[category] += 1
                if category == 'games':
                    value = normalize_game(row); keep(category, value['id'], value)
                elif category == 'advanced':
                    value = normalize_advanced(row); keep(category, (value['id'], value['team']), value)
                elif category == 'box':
                    for gid, team, stats in box_fragments(row):
                        key = (gid, team); previous = fragments.setdefault(key, {})
                        for stat, value in stats.items():
                            value = numeric(value) if numeric(value) is not None else value
                            if stat in previous and previous[stat] != value:
                                conflicts['box'].add(key); issues['box: conflicting duplicate records excluded'] += 1
                            else: previous[stat] = value
                elif category == 'lines':
                    for value in line_rows(row):
                        keep(category, (value['id'], value['provider']), value)
                elif category=='polls':
                    from poll_history import snapshots
                    values=[{k:v for k,v in row.items() if k!='_kind'}] if 'ranks' in row else snapshots([row])
                    for value in values:keep(category,value['id'],value)
                else: raise ValueError('Unrecognized export. Use Games, Team GAME stats, Advanced GAME stats or Betting Lines.')
                report['recognized'] += 1
            except (ValueError, TypeError, KeyError) as exc:
                report['rejected'] += 1; issues[str(exc)] += 1
        report['types'] = dict(counts); file_reports.append(report)
    for (gid, team), values in fragments.items():
        if (gid, team) in conflicts['box']: continue
        try: keep('box', (gid, team), normalize_box(gid, team, values))
        except ValueError as exc: issues[str(exc)] += 1
    dataset = dict(format=FORMAT, league='CFB', created_at=datetime.now(timezone.utc).isoformat(),
                   **{k: sorted(v.values(), key=lambda r: (r['id'], r.get('team', ''), r.get('provider', ''))) for k, v in maps.items()},
                   files=file_reports, issues=[dict(reason=k, count=v) for k, v in issues.items()])
    return dataset


def coverage(dataset):
    box = {(r['id'], r['team']) for r in dataset['box']}
    advanced = {(r['id'], r['team']) for r in dataset['advanced']}
    lines = {r['id'] for r in dataset['lines']}
    seasons = {}
    for game in dataset['games']:
        row = seasons.setdefault(game['season'], dict(season=game['season'], games=0, completed=0, both_box_scores=0, both_advanced=0, games_with_lines=0))
        row['games'] += 1; row['completed'] += bool(game['completed'])
        row['both_box_scores'] += all((game['id'], game[side]) in box for side in ('away', 'home'))
        row['both_advanced'] += all((game['id'], game[side]) in advanced for side in ('away', 'home'))
        row['games_with_lines'] += game['id'] in lines
    return [seasons[k] for k in sorted(seasons)]


def fetch_seasons(years, key, fetch=None, progress=None, include_lines=True):
    """Explicit, cached API download; return partial data on provider failure."""
    from data_pipeline import request_bytes, CFBD
    if not key or not key.strip():raise ValueError('Enter a CollegeFootballData API key first.')
    fetch=fetch or request_bytes
    years=sorted(set(int(y) for y in years))
    if not years or len(years)>20:raise ValueError('Select 1–20 seasons.')
    files=[];errors=[];calls=0
    headers={'Authorization':'Bearer '+key.strip()}
    def get(path,params):
        nonlocal calls
        content=fetch(CFBD+path,params,headers,ttl=365*86400);calls+=1
        value=json.loads(content)
        if not isinstance(value,list):raise ValueError('CFBD response must contain a list of records.')
        return content,value
    try:
        for year in years:
            content,games=get('/games',dict(year=year,seasonType='regular',classification='fbs'))
            files.append((f'{year}_games.json',content))
            weeks=sorted({int(g['week']) for g in games if g.get('week') is not None})
            content,_=get('/stats/game/advanced',dict(year=year,seasonType='regular'))
            files.append((f'{year}_advanced_game.json',content))
            content,_=get('/rankings',dict(year=year,seasonType='regular'))
            files.append((f'{year}_ap_rankings.json',content))
            if include_lines:
                content,_=get('/lines',dict(year=year,seasonType='regular'))
                files.append((f'{year}_lines.json',content))
            for week in weeks:
                content,_=get('/games/teams',dict(year=year,week=week,seasonType='regular'))
                files.append((f'{year}_week_{week}_team_game.json',content))
                if progress:progress(f'Loaded {year}, week {week}; {calls} requests including cached reads.')
    except (RuntimeError,ValueError,UnicodeError,TypeError,KeyError) as exc:
        errors.append('Download stopped; previously fetched files retained. '+str(exc))
    return files,dict(requests_including_cached_reads=calls,errors=errors)
