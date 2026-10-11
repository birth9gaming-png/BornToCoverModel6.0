"""Strict optional NFL provider import; missing data never becomes an estimate."""
from __future__ import annotations
import copy
import json
import math
from nfl_features_ext import utc
from nfl_layers import TEAM_FIELDS,settings
from nfl_stats import code
import pandas as pd

INPUT_KEYS={'team_metrics':'nfl_team_inputs','weather':'nfl_weather_inputs',
            'players':'nfl_player_candidates','player_games':'nfl_player_game_logs','attd_quotes':'nfl_attd_quotes'}


def _finite(value,key,low=None,high=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (low is not None and value<low) or (high is not None and value>high):raise ValueError('Invalid NFL input: '+key)


def _time(value,key):
    if pd.isna(utc(value)):raise ValueError(key+' needs a timestamp with timezone.')


def merge_inputs(data,payload):
    """Return a checked copy. File import and live source adapters share the schema."""
    if not isinstance(payload,dict) or payload.get('format')!='nfl-layer-inputs-v1':raise ValueError('Expected nfl-layer-inputs-v1 JSON. See the included schema/example.')
    updated=copy.deepcopy(data)
    if 'settings' in payload:
        updated['nfl_layer_settings']={**settings(data),**payload['settings']};settings(updated)
    for category,destination in INPUT_KEYS.items():
        if category not in payload:continue
        rows=payload[category]
        if not isinstance(rows,list) or len(rows)>50000:raise ValueError(category+' must be a list with at most 50,000 rows.')
        checked=[]
        for original in rows:
            if not isinstance(original,dict):raise ValueError('NFL input rows must be objects.')
            row=copy.deepcopy(original)
            if 'team' in row:row['team']=code(row['team'])
            if category=='team_metrics':
                if not row.get('team'):raise ValueError('Team metrics require team code.')
                for key in ('season','through_week'):
                    _finite(row.get(key),key,0)
                    if row[key]!=int(row[key]):raise ValueError(key+' must be an integer.')
                _time(row.get('as_of'),'Metric as_of')
                for key in TEAM_FIELDS:
                    if row.get(key) is None:continue
                    _finite(row[key],key,0 if key=='non_blitz_pressure_rate' or 'line_yards' in key else -200,1 if key=='non_blitz_pressure_rate' else 20 if 'line_yards' in key else 200)
            elif category=='weather':
                if not row.get('game_id'):raise ValueError('Weather requires game_id.')
                _finite(row.get('forecast_wind_mph'),'forecast_wind_mph',0,150);_time(row.get('weather_as_of'),'Weather as_of')
                if row.get('roof') not in ('outdoors','open','dome','closed','indoors','unknown'):raise ValueError('Weather roof must describe game exposure.')
            elif category=='players':
                if not row.get('player_id') or not row.get('team') or not row.get('position'):raise ValueError('Players need player_id, team and position.')
                rank=row.get('depth_rank',row.get('usage_rank'))
                _finite(rank,'depth rank',1,10)
                if rank!=int(rank):raise ValueError('Depth rank must be an integer.')
                for key in ('projected_opportunities','projected_xTD'):
                    if row.get(key) is not None:_finite(row[key],key,0)
                if row.get('recency_verified') is not None and not isinstance(row['recency_verified'],bool):raise ValueError('recency_verified must be Boolean.')
            elif category=='player_games':
                if not row.get('player_id') or not row.get('game_id'):raise ValueError('Player games need player_id and game_id.')
                if not isinstance(row.get('active'),bool):raise ValueError('Player game active must be Boolean.')
                _time(row.get('kickoff'),'Player game kickoff')
                _finite(row.get('touchdowns'),'touchdowns',0)
                if row['touchdowns']!=int(row['touchdowns']):raise ValueError('TD counts must be integers.')
                for key in ('carries','targets','xTD','redzone_targets','redzone_touches','team_redzone_targets','team_redzone_touches'):
                    if row.get(key) is not None:_finite(row[key],key,0)
            elif category=='attd_quotes':
                if not all(row.get(k) for k in ('player_id','game_id','book')):raise ValueError('ATTD quotes need player_id, game_id and book.')
                _finite(row.get('odds'),'ATTD odds')
                if abs(row['odds'])<100:raise ValueError('ATTD odds must be American odds.')
                _time(row.get('quoted_at'),'ATTD quote time')
            checked.append(row)
        keys={'team_metrics':('team','season','through_week','as_of'),'weather':('game_id','weather_as_of'),
              'players':('player_id',),'player_games':('player_id','game_id'),'attd_quotes':('game_id','player_id','book','quoted_at')}[category]
        merged={tuple(str(r.get(k)) for k in keys):r for r in updated.get(destination,[])}
        for row in checked:merged[tuple(str(row.get(k)) for k in keys)]=row
        updated[destination]=list(merged.values())
    return updated


def export_inputs(data):
    return json.dumps(dict(format='nfl-layer-inputs-v1',settings=settings(data),**{k:data.get(v,[]) for k,v in INPUT_KEYS.items()}),allow_nan=False,indent=2)
