"""Stateless expected-score prototype; coefficients are editable priors."""
from __future__ import annotations
import copy
import math
from statistics import NormalDist
from matchup_history import HISTORY_DEFAULTS, adjustments as history_adjustments
from schedule_context import defaults as context_defaults, adjustments as context_adjustments, validate_context
from power_ratings import defaults as power_defaults, comparison as power_comparison
from ats_history import DEFAULTS as ATS_DEFAULTS, comparison as ats_comparison
from cfb_ml import adjustment as ml_adjustment
from nfl_ml import DEFAULTS as NFL_DEFAULTS, adjustment as nfl_ml_adjustment, scenarios as nfl_scenarios
from nfl_stats import NFL_METRICS

NORMAL = NormalDist()
METRICS = ('y', 'dy', 's', 'ds', 'x', 'dx', 'p', 't')
CFB_METRICS = ('ofei', 'dfei', 'returning_starters_off', 'returning_starters_def',
               'portal_net_rating', 'roster_games_played')
# Editable instructional priors; these are not fitted historical coefficients.
CFB_DEFAULTS = dict(fei_weight=.25, fei_drives=12., starter_weight=.20,offense_share=.5,
                    starters_reference=6., portal_weight=1., portal_scale=5.,
                    roster_half_life=4.)
LEAGUE_CONTROLS = ('by','bs','shrink','turnover_retention','turnover_point_value')


def parameters_for_league(parameters, league):
    return {**parameters, **{k: parameters[league].get(k, parameters[k]) for k in LEAGUE_CONTROLS}}


def ensure_cfb_defaults(data):
    for key, value in CFB_DEFAULTS.items():
        data['parameters']['CFB'].setdefault(key, value)
    for team in data['teams'].values():
        if team.get('league') == 'CFB':
            for key in CFB_METRICS:
                team.setdefault(key, None)
    for league in ('NFL','CFB'):
        for key,value in HISTORY_DEFAULTS.items(): data['parameters'][league].setdefault(key,value)
        for key,value in context_defaults(league).items(): data['parameters'][league].setdefault(key,value)
        for key,value in power_defaults(league).items(): data['parameters'][league].setdefault(key,value)
        for key,value in ATS_DEFAULTS.items():data['parameters'][league].setdefault(key,value)
    data['parameters']['CFB'].setdefault('power_method','massey')
    data['parameters']['CFB'].setdefault('ml_weight',1.)
    data['parameters']['CFB'].setdefault('context_ranked_only',False)
    data['parameters']['CFB'].setdefault('selection_style','standard')
    for key,value in NFL_DEFAULTS.items():data['parameters']['NFL'].setdefault(key,value)
    return data


def snapshot_season(data):
    if data.get('season') is not None:
        return int(data['season'])
    return int(str(data.get('snapshot_date', data.get('as_of', '')))[:4]) if str(data.get('snapshot_date', data.get('as_of', '')))[:4].isdigit() else None


def cfb_adjustments(data, game, away, home, params, pace):
    par = {**CFB_DEFAULTS, **params['CFB']}
    notes = []
    season = snapshot_season(data)
    cutoff = game.get('feature_through_week')
    fei_ok = True
    for team in (away, home):
        name = team.get('name', 'Team')
        if any(team.get(k) is None for k in ('ofei', 'dfei')):
            fei_ok = False; notes.append(name + ': FEI missing; the game uses the original efficiency forecast.')
        elif season is not None and team.get('fei_season') != season:
            fei_ok = False; notes.append(name + ': FEI season missing or mismatched; FEI skipped.')
        elif team.get('fei_through_week') is None:
            fei_ok = False; notes.append(name + ': FEI through-week missing; FEI skipped.')
        elif cutoff is not None and team['fei_through_week'] > cutoff:
            fei_ok = False; notes.append(name + ': FEI includes weeks after the feature cutoff; FEI skipped.')
    decays = {}
    for side, team in (('away', away), ('home', home)):
        games = team.get('roster_games_played')
        if games is None: games = team.get('sample_games')
        if season is not None and team.get('roster_season') != season:
            decays[side] = None
            notes.append(team.get('name', side) + ': roster season missing or mismatched; roster adjustments skipped.')
        elif games is None:
            decays[side] = None
            notes.append(team.get('name', side) + ': games played missing; roster adjustments skipped.')
        else:
            decays[side] = 2 ** (-float(games) / par['roster_half_life'])
    portal = 0.
    portal_ok = all(t.get('portal_net_rating') is not None for t in (away, home)) and all(v is not None for v in decays.values())
    if portal_ok:
        portal = par['portal_weight'] * (decays['away'] * away['portal_net_rating'] - decays['home'] * home['portal_net_rating']) / (2 * par['portal_scale'])
    else:
        notes.append('Portal comparison needs both net ratings and games played; portal adjustment skipped.')
    return dict(fei_active=fei_ok, fei_weight=par['fei_weight'] if fei_ok else 0.,
                expected_drives=pace * par['fei_drives'], roster_decay=decays,
                portal_active=portal_ok, portal_away=portal, notes=notes, parameters=par)


def validate(data):
    if not isinstance(data, dict) or not isinstance(data.get('teams'), dict) or not isinstance(data.get('games'), list):
        raise ValueError('Snapshot needs teams (object) and games (list).')
    par = data.get('parameters', {})
    for key in ('by', 'bs', 'shrink', 'turnover_retention', 'turnover_point_value'):
        if not isinstance(par.get(key), (int, float)) or not math.isfinite(par[key]):
            raise ValueError(f'Missing or invalid parameter: {key}')
    for league in ('NFL', 'CFB'):
        for key in LEAGUE_CONTROLS:
            value=par.get(league,{}).get(key,par.get(key))
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Invalid league control: '+league+'.'+key)
        if par[league].get('power_method','massey' if league=='CFB' else 'ridge') not in ('massey','ridge'):
            raise ValueError('Unknown result-rating method.')
        mw=par[league].get('ml_weight',1.)
        if isinstance(mw,bool) or not isinstance(mw,(int,float)) or not math.isfinite(mw) or not 0<=mw<=1:
            raise ValueError('Gradient-boosting blend must be 0–1.')
        for key in ('q', 'n', 'y', 's', 'x', 'bx', 'h', 'sigma_spread', 'sigma_total'):
            if not isinstance(par.get(league, {}).get(key), (int, float)) or not math.isfinite(par[league][key]):
                raise ValueError(f'Missing or invalid parameter: {league}.{key}')
        if min(par[league]['n'], par[league]['sigma_spread'], par[league]['sigma_total']) <= 0:
            raise ValueError('Reference plays and residual sigmas must be positive.')
        hp={**HISTORY_DEFAULTS,**par[league]}
        if any(isinstance(hp[k],bool) or not isinstance(hp[k],(int,float)) or not math.isfinite(hp[k]) for k in HISTORY_DEFAULTS):
            raise ValueError('Invalid history/calibration parameter.')
        if not all(0<=hp[k]<=1 for k in ('h2h_margin_weight','h2h_total_weight')) or min(hp[k] for k in ('h2h_half_life_years','h2h_prior_games','h2h_difference_cap','margin_scale','total_scale'))<=0:
            raise ValueError('Invalid history weights, scales or shrinkage.')
        pp={**power_defaults(league),**par[league]}
        if any(isinstance(pp[k],bool) or not isinstance(pp[k],(int,float)) or not math.isfinite(pp[k]) for k in power_defaults(league)) or not 0<=pp['power_weight']<=1 or pp['power_ridge']<=0 or pp['power_min_games']<1 or pp['power_min_games']!=int(pp['power_min_games']):
            raise ValueError('Result-rating weight must be 0–1, shrinkage positive, and minimum games a positive integer.')
        ap={**ATS_DEFAULTS,**par[league]}
        if any(isinstance(ap[k],bool) or not isinstance(ap[k],(int,float)) or not math.isfinite(ap[k]) for k in ATS_DEFAULTS) or min(ap['ats_weight'],ap['ats_cap_points'])<0 or ap['ats_prior_games']<=0:
            raise ValueError('ATS weights/cap must be nonnegative and sample-size caution positive.')
    cfb = {**CFB_DEFAULTS, **par['CFB']}
    nfl={**NFL_DEFAULTS,**par['NFL']}
    for key in NFL_DEFAULTS:
        if isinstance(nfl[key],bool) or not isinstance(nfl[key],(int,float)) or not math.isfinite(nfl[key]):
            raise ValueError('Invalid NFL parameter: '+key)
    if not 0<=nfl['epa_weight']<=1 or nfl['rest_cap_points']<0 or abs(nfl['rest_day_weight'])>2:
        raise ValueError('NFL EPA blend must be 0–1, rest cap nonnegative, and rest weight between −2 and 2.')
    for key in CFB_DEFAULTS:
        if isinstance(cfb[key], bool) or not isinstance(cfb[key], (int, float)) or not math.isfinite(cfb[key]):
            raise ValueError('Invalid CFB parameter: ' + key)
    if not 0 <= cfb['fei_weight'] <= 1 or not 0 <= cfb['starters_reference'] <= 11 or not 0<=cfb['offense_share']<=1:
        raise ValueError('FEI weight must be 0–1; starter reference must be 0–11.')
    if min(cfb['fei_drives'], cfb['portal_scale'], cfb['roster_half_life']) <= 0 or min(cfb['starter_weight'], cfb['portal_weight']) < 0:
        raise ValueError('CFB scales and half-life must be positive; roster weights cannot be negative.')
    for name, team in data['teams'].items():
        if team.get('league') not in ('NFL', 'CFB'):
            raise ValueError(f'Invalid league for {name}')
        for key in METRICS:
            if team.get(key) is not None and (not isinstance(team[key], (int, float)) or not math.isfinite(team[key])):
                raise ValueError(f'Invalid team metric {name}.{key}')
        if team.get('league')=='NFL':
            for key in NFL_METRICS:
                if team.get(key) is not None and (isinstance(team[key],bool) or not isinstance(team[key],(int,float)) or not math.isfinite(team[key])):
                    raise ValueError(f'Invalid NFL team metric {name}.{key}')
        for key in CFB_METRICS + ('fei_season', 'fei_through_week', 'roster_season'):
            value = team.get(key)
            if value is None: continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid team metric {name}.{key}')
            if key in ('returning_starters_off', 'returning_starters_def') and (not 0 <= value <= 11 or value != int(value)):
                raise ValueError('Returning starters must be an integer from 0 to 11.')
            if key in ('roster_games_played', 'fei_through_week', 'fei_season', 'roster_season') and (value < 0 or value != int(value)):
                raise ValueError(key + ' must be a nonnegative integer.')
    for game in data['games']:
        if not isinstance(game, dict) or not all(isinstance(game.get(k), str) for k in ('away', 'home')):
            raise ValueError('Each game needs away and home team identifiers.')
        for key in ('line_away','total','odds_away','odds_home','odds_over','odds_under','injury_away','injury_home','rest_margin_home','weather_total'):
            if game.get(key) is not None and (not isinstance(game[key], (int, float)) or not math.isfinite(game[key])):
                raise ValueError(f'Invalid game value: {key}')
    validate_context(data)
    return data


def quotes_for(game):
    if 'quotes' in game:
        return game['quotes']
    quotes = []
    for kind, selection, field in [('spread', 'away', 'line_away'), ('spread', 'home', 'line_away'), ('total', 'over', 'total'), ('total', 'under', 'total')]:
        line, odds = game.get(field), game.get('odds_' + selection)
        if line is None or odds is None:
            continue
        quotes.append(dict(kind=kind, selection=selection, line=-line if selection == 'home' else line,
                           odds=odds, book=game.get('book', 'Imported'), updated_at=game.get('odds_updated_at')))
    return quotes


def probabilities(mean, sigma, threshold, greater):
    if sigma <= 0 or not math.isfinite(threshold) or not math.isclose(2 * threshold, round(2 * threshold)):
        raise ValueError('Use positive sigma and integer/half-point market lines.')
    if math.isclose(threshold, round(threshold), abs_tol=1e-10):
        lo = NORMAL.cdf((threshold - .5 - mean) / sigma)
        hi = NORMAL.cdf((threshold + .5 - mean) / sigma)
        return (1 - hi, hi - lo, lo) if greater else (lo, hi - lo, 1 - hi)
    below = NORMAL.cdf((threshold - mean) / sigma)
    return (1 - below, 0, below) if greater else (below, 0, 1 - below)


def evaluate(quote, result, par):
    q = copy.deepcopy(quote)
    odds, line = float(q['odds']), float(q['line'])
    if not math.isfinite(odds) or abs(odds) < 100:
        raise ValueError('American odds require absolute value at least 100.')
    selection, kind = q['selection'], q['kind']
    if kind == 'spread' and selection in ('away', 'home'):
        mean = result['margin_away'] if selection == 'away' else -result['margin_away']
        threshold, greater, edge, sigma = -line, True, mean + line, par['sigma_spread']
        label = result['game'][selection] + f' {line:+g}'
    elif kind == 'total' and selection in ('over', 'under'):
        mean, threshold, greater, sigma = result['fair_total'], line, selection == 'over', par['sigma_total']
        edge = (mean - line) * (1 if greater else -1)
        label = selection.title() + f' {line:g}'
    else:
        raise ValueError('Unsupported market or selection.')
    win, push, loss = probabilities(mean, sigma, threshold, greater)
    profit = 100 / abs(odds) if odds < 0 else odds / 100
    q.update(label=label, edge=edge, p_win=win, p_push=push, p_loss=loss,
             profit_multiple=profit, conditional_ev=profit * win - loss)
    return q


def project(data, game, parameters=None):
    params = data['parameters'] if parameters is None else parameters
    away, home = data['teams'].get(game['away']), data['teams'].get(game['home'])
    missing = [game[side] for side, team in [('away', away), ('home', home)]
               if not team or any(team.get(k) is None for k in METRICS)]
    if missing:
        return dict(game=copy.deepcopy(game), league=game.get('league') or (away or home or {}).get('league'),
                    status='Missing statistics', missing=missing, markets=[], best_market=None)
    league = away['league']
    params = parameters_for_league(params, league)
    if home['league'] != league:
        raise ValueError('Teams must belong to the same league.')
    par = params[league]
    if league=='NFL':par={**NFL_DEFAULTS,**par}
    pace = (away['p'] + home['p']) / (2 * par['n'])
    venue = 0 if game.get('neutral', False) else par['h']
    rest = game.get('rest_margin_home', 0)
    turn = params['turnover_point_value'] * params['turnover_retention'] * (away['t'] - home['t']) / 4
    output = dict(game=copy.deepcopy(game), league=league, pace=pace, status='Projected')
    extras = cfb_adjustments(data, game, away, home, params, pace) if league == 'CFB' else None
    if extras is not None: output['cfb_extensions'] = extras
    nfl_extra=nfl_scenarios(game,away,home,par,pace) if league=='NFL' else None
    if nfl_extra is not None:
        output['nfl_extensions']=nfl_extra
        rest+=nfl_extra['rest_margin_home']
    for side, own, opp, sign in [('away', away, home, -1), ('home', home, away, 1)]:
        offensive_share=par.get('offense_share',.5) if league=='CFB' else .5
        matched = {k: offensive_share*own[k] + (1-offensive_share)*opp['d' + k] for k in ('y', 's', 'x')}
        components = dict(ypp=params['by'] * (matched['y'] - par['y']),
                          success=params['bs'] * (matched['s'] - par['s']),
                          explosiveness=par['bx'] * (matched['x'] - par['x']))
        base = par['q'] + params['shrink'] * sum(components.values())
        parts = dict(pace_base=pace * base, turnover=-sign * turn,
                     venue=sign * venue / 2, rest=sign * rest / 2,
                     injury=game.get('injury_' + side, 0), weather=game.get('weather_total', 0) / 2)
        if nfl_extra is not None:
            epa_score=nfl_extra['epa_scores'][side]
            parts['nfl_epa']=nfl_extra['epa_weight']*(epa_score-pace*base) if epa_score is not None else 0.
        if extras is not None:
            fei_score = pace * par['q'] + extras['expected_drives'] * (own['ofei'] - opp['dfei']) if extras['fei_active'] else None
            parts['fei_blend'] = extras['fei_weight'] * (fei_score - pace * base) if fei_score is not None else 0.
            opposite = 'home' if side == 'away' else 'away'
            continuity = 0.
            for role, team, decay, direction in [('off', own, extras['roster_decay'][side], 1), ('def', opp, extras['roster_decay'][opposite], -1)]:
                count = team.get('returning_starters_' + role)
                if count is not None and decay is not None:
                    continuity += direction * decay * extras['parameters']['starter_weight'] * (count - extras['parameters']['starters_reference'])
                elif count is None:
                    extras['notes'].append(team.get('name', role) + ': returning ' + role + ' starters missing; that term skipped.')
            parts['returning_starters'] = continuity
            parts['portal'] = -sign * extras['portal_away']
        score = sum(parts.values())
        if not math.isfinite(score) or score <= 0:
            raise ValueError('Inputs produce a nonpositive score. Revise the assumptions.')
        output[side] = dict(matched=matched, feature_contributions=components,
                            base=base, score_parts=parts, score=score)
        if extras is not None: output[side]['fei_score'] = fei_score
    efficiency_margin=sum(output['away']['score_parts'].get(k,0)-output['home']['score_parts'].get(k,0) for k in ('pace_base','fei_blend','nfl_epa'))
    scenario_data={**data,'parameters':params,'season':snapshot_season(data)}
    power=power_comparison(scenario_data,{**game,'league':league},params,efficiency_margin)
    output['power_rating']=power
    for side,sign in (('away',1),('home',-1)):
        output[side]['score_parts']['power_rating']=sign*power['margin_adjustment']/2
        output[side]['score']=sum(output[side]['score_parts'].values())
    ml=(ml_adjustment if league=='CFB' else nfl_ml_adjustment)(data,{**game,'league':league},params,output)
    output['gradient_boosting']=ml
    for side,sign in (('away',1),('home',-1)):
        output[side]['score_parts']['gradient_boosting']=sign*ml['margin_adjustment']/2+ml['total_adjustment']/2
        output[side]['score']=sum(output[side]['score_parts'].values())
    context=context_adjustments(scenario_data,{**game,'league':league})
    output['schedule_context']=context
    for side,sign in (('away',1),('home',-1)):
        output[side]['score_parts']['schedule_context']=sign*context['margin_adjustment']/2+context['total_adjustment']/2
        output[side]['score']=sum(output[side]['score_parts'].values())
    ats=ats_comparison(data,{**game,'league':league},params)
    output['ats_history']=ats
    for side,sign in (('away',1),('home',-1)):
        output[side]['score_parts']['ats_history']=sign*ats['margin_adjustment']/2
        output[side]['score']=sum(output[side]['score_parts'].values())
    margin = output['away']['score'] - output['home']['score']
    total = output['away']['score'] + output['home']['score']
    output['raw_margin_away']=margin;output['raw_total']=total
    calibration={**HISTORY_DEFAULTS,**par}
    # Calibration acts on team strength; home/rest/injury/weather modifiers retain
    # their original point units and are not multiplied by a strength slope.
    margin_modifiers=sum(output['away']['score_parts'][k]-output['home']['score_parts'][k] for k in ('venue','rest','injury','weather','schedule_context','ats_history'))
    total_modifiers=sum(output['away']['score_parts'][k]+output['home']['score_parts'][k] for k in ('venue','rest','injury','weather','schedule_context','ats_history'))
    cm=(calibration['margin_scale']-1)*(margin-margin_modifiers)+calibration['margin_intercept']
    ct=(calibration['total_scale']-1)*(total-total_modifiers)+calibration['total_intercept']
    dm,dt,summary=history_adjustments(scenario_data,{**game,'league':league},margin+cm,total+ct,venue,rest)
    output['head_to_head']=summary
    for side,sign in (('away',1),('home',-1)):
        output[side]['score_parts']['calibration']=sign*cm/2+ct/2
        output[side]['score_parts']['head_to_head']=sign*dm/2+dt/2
        output[side]['score']=sum(output[side]['score_parts'].values())
        if not math.isfinite(output[side]['score']) or output[side]['score']<=0:
            raise ValueError('History/calibration produces a nonpositive score; revise its weights.')
    margin=output['away']['score']-output['home']['score']
    total=output['away']['score']+output['home']['score']
    output.update(margin_away=margin, fair_line_away=-margin, fair_total=total)
    output['spread_edge_away'] = margin + game['line_away'] if game.get('line_away') is not None else None
    output['total_edge_over'] = total - game['total'] if game.get('total') is not None else None
    output['markets'], output['quote_errors'] = [], []
    for quote in quotes_for(game):
        try:
            output['markets'].append(evaluate(quote, output, par))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            output['quote_errors'].append(str(exc))
    output['best_market'] = max(output['markets'], key=lambda x: x['conditional_ev'], default=None)
    if not output['markets']:
        output['status'] = 'Projection only — no usable odds'
    return output


def run(data, parameters=None):
    rows = []
    for game in data['games']:
        try:
            rows.append(project(data, game, parameters))
        except (ValueError, ArithmeticError, TypeError, KeyError, ZeroDivisionError) as exc:
            rows.append(dict(game=copy.deepcopy(game), league=game.get('league'), status=str(exc), markets=[], best_market=None))
    return rows
