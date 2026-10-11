"""Optional NFL source adapters, called only by an explicit slate refresh."""
from __future__ import annotations
import io
import json
from collections import defaultdict
import pandas as pd
from nfl_attd import build_player_inputs,name_key
from nfl_features_ext import utc
from nfl_stats import code

BASE='https://github.com/nflverse/nflverse-data/releases/download/'
HOME_LOCATIONS={
    'ARI':('Glendale','Arizona'),'ATL':('Atlanta','Georgia'),'BAL':('Baltimore','Maryland'),
    'BUF':('Orchard Park','New York'),'CAR':('Charlotte','North Carolina'),'CHI':('Chicago','Illinois'),
    'CIN':('Cincinnati','Ohio'),'CLE':('Cleveland','Ohio'),'DAL':('Arlington','Texas'),
    'DEN':('Denver','Colorado'),'DET':('Detroit','Michigan'),'GB':('Green Bay','Wisconsin'),
    'HOU':('Houston','Texas'),'IND':('Indianapolis','Indiana'),'JAX':('Jacksonville','Florida'),
    'KC':('Kansas City','Missouri'),'LA':('Inglewood','California'),'LAC':('Inglewood','California'),
    'LV':('Paradise','Nevada'),'MIA':('Miami Gardens','Florida'),'MIN':('Minneapolis','Minnesota'),
    'NE':('Foxborough','Massachusetts'),'NO':('New Orleans','Louisiana'),'NYG':('East Rutherford','New Jersey'),
    'NYJ':('East Rutherford','New Jersey'),'PHI':('Philadelphia','Pennsylvania'),'PIT':('Pittsburgh','Pennsylvania'),
    'SEA':('Seattle','Washington'),'SF':('Santa Clara','California'),'TB':('Tampa','Florida'),
    'TEN':('Nashville','Tennessee'),'WAS':('Landover','Maryland')}


def fetch_wind_forecasts(games,fetch,config,as_of):
    """Hourly sustained 10m wind in MPH; never target-game realized weather.

    Home-city forecasts are approximate venue forecasts. Neutral/international
    sites and unknown/retractable roof exposure require a specific import.
    """
    if not config['enabled'] or not config['weather_enabled'] or not config['fetch_weather_data']:return [],[]
    now=utc(as_of);forecasts=[];notes=[]
    for game in games:
        kickoff=utc(game.get('kickoff'));location=HOME_LOCATIONS.get(code(game.get('home')))
        roof=str(game.get('roof','unknown')).lower()
        if game.get('neutral') or roof not in ('outdoors','open','outdoor') or not location or pd.isna(kickoff) or not now<kickoff<now+pd.Timedelta(days=15):continue
        try:
            city,state=location
            geo=json.loads(fetch('https://geocoding-api.open-meteo.com/v1/search',dict(name=city,count=10,language='en',format='json',countryCode='US'),ttl=86400))
            matches=[r for r in geo.get('results',[]) if r.get('country_code')=='US' and r.get('admin1')==state and name_key(r.get('name'))==name_key(city)]
            if not matches:raise ValueError('Venue city could not be resolved.')
            # Exact city/state match excludes airports and similarly named
            # neighborhoods. Use the principal populated city when duplicated.
            place=max(matches,key=lambda r:r.get('population') or 0)
            if len(matches)>1 and sum((r.get('population') or 0)==(place.get('population') or 0) for r in matches)>1:raise ValueError('Venue city is ambiguous.')
            result=json.loads(fetch('https://api.open-meteo.com/v1/forecast',dict(latitude=place['latitude'],longitude=place['longitude'],hourly='wind_speed_10m',wind_speed_unit='mph',timezone='UTC',timeformat='unixtime',forecast_days=16),ttl=1800))
            hourly=result['hourly'];times=pd.to_datetime(hourly['time'],unit='s',utc=True)
            distance=abs(times-kickoff);index=int(distance.argmin())
            wind=float(hourly['wind_speed_10m'][index])
            if distance[index]>pd.Timedelta(hours=1) or not 0<=wind<=150:raise ValueError('Forecast does not cover kickoff.')
            forecasts.append(dict(game_id=str(game['id']),forecast_wind_mph=wind,weather_as_of=as_of,forecast_hour=times[index].isoformat(),roof='outdoors' if roof=='outdoor' else roof,
                source='Open-Meteo hourly sustained 10m wind; approximate '+city+', '+state+' venue location'))
        except (RuntimeError,ValueError,KeyError,TypeError,OSError):notes.append('Sustained-wind forecast unavailable for '+game['away']+' @ '+game['home']+'.')
    return forecasts,notes


def player_bundle(pbp,schedule,season,through_week,fetch,config):
    notes=[];logs=[];players=[];injury_players=[]
    if config['enabled'] and config['attd_enabled'] and config['fetch_player_data']:
        try:
            roster=pd.read_csv(io.BytesIO(fetch(BASE+f'rosters/roster_{season}.csv',ttl=3600)),low_memory=False)
            try:snaps=pd.read_csv(io.BytesIO(fetch(BASE+f'snap_counts/snap_counts_{season}.csv',ttl=3600)),low_memory=False)
            except (RuntimeError,ValueError,KeyError,OSError):
                snaps=pd.DataFrame();notes.append('Actual NFL snap feed unavailable; automatic ATTD recency rules stay off until active-game coverage is complete.')
            logs,players=build_player_inputs(pbp,schedule,roster,snaps,through_week)
            injury_players=injury_baselines(roster,snaps,through_week,schedule)
            notes.append(f'Automatic player inputs: {len(players)} skill players and {len(logs)} prior active-game logs. xTD is a labeled PBP proxy.')
        except (RuntimeError,ValueError,KeyError,OSError) as exc:notes.append('NFL player inputs unavailable: '+str(exc))
    return dict(nfl_player_game_logs=logs,nfl_player_candidates=players,nfl_injury_players=injury_players,nfl_layer_feed_notes=notes)


def injury_baselines(roster,snaps,through_week,schedule=None):
    """Estimate existing absences/role shares from complete recent team snap sheets.

    No player value is learned here. A missing snap sheet never means inactive.
    """
    required={'team','pfr_player_id','game_id','week','offense_snaps','defense_snaps'}
    if not required.issubset(snaps) or not {'team','pfr_id','full_name','position'}.issubset(roster):return []
    snap=snaps[pd.to_numeric(snaps.week,errors='coerce').le(through_week)].copy()
    snap['team']=snap.team.map(code)
    output=[]
    for team,group in snap.groupby('team'):
        last=group[['game_id','week']].drop_duplicates().sort_values('week',ascending=False).head(4)
        gids=list(last.game_id);part=group[group.game_id.isin(gids)]
        # At least two recent team games are needed to identify an existing absence.
        if len(gids)<2 or last.week.max()<through_week-1:continue
        if schedule is not None:
            expected=pd.DataFrame(schedule)
            required_schedule={'game_id','week','home_team','away_team','home_score','away_score'}
            if not required_schedule.issubset(expected):continue
            expected=expected[(expected.home_team.map(code).eq(team)|expected.away_team.map(code).eq(team)) &
                              pd.to_numeric(expected.week,errors='coerce').le(through_week) & expected.home_score.notna() & expected.away_score.notna()]
            ids=set(expected.sort_values('week',ascending=False).head(4).game_id)
            if not ids or ids!=set(gids):continue
        people=roster[roster.team.map(code).eq(team)].drop_duplicates('pfr_id',keep='last')
        for _,person in people.iterrows():
            pid=person.get('pfr_id')
            if pd.isna(pid):continue
            rows=part[part.pfr_player_id.eq(pid)].copy();position=str(person.position)
            field='defense' if position in ('DL','DT','DE','EDGE','LB','ILB','OLB','CB','DB','S','FS','SS') else 'offense'
            active=pd.to_numeric(rows.get(field+'_snaps',pd.Series(dtype=float)),errors='coerce').fillna(0)>0
            played=rows[active];share=None
            pct=field+'_pct'
            if pct in played and not played.empty:
                values=pd.to_numeric(played[pct],errors='coerce').dropna()
                if not values.empty:share=float(values.mean());share=share/100 if share>1 else share
            output.append(dict(name=str(person.full_name),team=team,position=position,
                player_id=str(person.get('gsis_id','')),baseline_verified=share is not None and 0<=share<=1,
                baseline_absence=1-played.game_id.nunique()/len(gids),role_share=share if share is not None else 0.,
                recent_team_games=len(gids),through_week=int(last.week.max())))
    return output


def add_pressure_charting(pbp,season,fetch,config):
    """Join real pressure and blitz tags by game/play; never infer them from sacks."""
    if not config['enabled'] or not config['trenches_enabled'] or not config['fetch_pressure_data'] or pbp.empty:return pbp,[]
    notes=[];result=pbp.copy()
    if 'play_id' not in result:return result,['Play IDs missing; pressure charting cannot be joined.']
    if 'was_pressure' not in result or not result.was_pressure.notna().any():
        result=result.drop(columns='was_pressure',errors='ignore')
        try:
            participation=pd.read_csv(io.BytesIO(fetch(BASE+f'pbp_participation/pbp_participation_{season}.csv',ttl=3600)),usecols=lambda c:c in ('nflverse_game_id','game_id','play_id','was_pressure'),low_memory=False)
            participation=participation.rename(columns={'nflverse_game_id':'game_id'})
            if not {'game_id','play_id','was_pressure'}.issubset(participation):raise ValueError('Pressure tags missing from participation feed.')
            result=result.merge(participation[['game_id','play_id','was_pressure']],on=['game_id','play_id'],how='left',validate='one_to_one')
        except (RuntimeError,ValueError,KeyError,OSError):return result,['Actual pressure tags unavailable for this season; non-blitz pressure term needs a provider input.']
    has_blitz=any(k in result and result[k].notna().any() for k in ('n_blitzers','is_blitz'))
    if not has_blitz:
        result=result.drop(columns=['n_blitzers','is_blitz'],errors='ignore')
        try:
            chart=pd.read_csv(io.BytesIO(fetch(BASE+f'ftn_charting/ftn_charting_{season}.csv',ttl=3600)),usecols=lambda c:c in ('nflverse_game_id','nflverse_play_id','n_blitzers'),low_memory=False).rename(columns={'nflverse_game_id':'game_id','nflverse_play_id':'play_id'})
            if not {'game_id','play_id','n_blitzers'}.issubset(chart):raise ValueError('Blitz tags missing.')
            result=result.merge(chart[['game_id','play_id','n_blitzers']],on=['game_id','play_id'],how='left',validate='one_to_one')
        except (RuntimeError,ValueError,KeyError,OSError):notes.append('Blitz tags unavailable; pressure alone is insufficient for non-blitz pressure rate.')
    return result,notes


def fetch_attd_quotes(games,players,key,regions,fetch,config,as_of):
    """Use an event-level ATTD market; require an unambiguous full-name match."""
    if not config['enabled'] or not config['attd_enabled'] or not config['fetch_attd_odds'] or not key:return [],[]
    by_name=defaultdict(list)
    for player in players:by_name[(code(player['team']),name_key(player.get('name')))].append(str(player['player_id']))
    quotes=[];notes=[];now=utc(as_of)
    for g in games:
        kickoff=utc(g.get('kickoff'))
        if g.get('league')!='NFL' or not g.get('odds_event_id') or g.get('total') is None or g['total']<config['attd_min_total'] or pd.isna(kickoff) or kickoff<=now:continue
        try:
            url='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/'+str(g['odds_event_id'])+'/odds'
            event=json.loads(fetch(url,dict(apiKey=key,regions=regions,markets='player_anytime_td',oddsFormat='american'),ttl=120))
            for book in event.get('bookmakers',[]):
                for market in book.get('markets',[]):
                    if market.get('key')!='player_anytime_td':continue
                    for outcome in market.get('outcomes',[]):
                        if str(outcome.get('name','')).lower() not in ('yes','over'):continue
                        name=name_key(outcome.get('description'));matches=set()
                        for side in ('away','home'):matches.update(by_name.get((code(g[side]),name),[]))
                        if len(matches)!=1:continue
                        odds=outcome.get('price')
                        if not isinstance(odds,(int,float)) or abs(odds)<100:continue
                        quotes.append(dict(game_id=str(g['id']),player_id=matches.pop(),book=book.get('title',book.get('key')),odds=odds,quoted_at=market.get('last_update',book.get('last_update'))))
        except (RuntimeError,ValueError,KeyError,OSError):notes.append('ATTD quotes unavailable for '+g['away']+' @ '+g['home']+'. Check provider market coverage.')
    return quotes,notes
