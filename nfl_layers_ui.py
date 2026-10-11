"""Controls live only inside the NFL tab; ATTD is a separate section."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from workspace_runtime import rerun
from nfl_layers import DEFAULTS,settings,forecast_as_of,eligible_team_metrics
from nfl_attd import rate_anytime_touchdowns
from auto_injuries import eligible_reports,settings as injury_settings
from datetime import datetime,timezone
from nfl_inputs import merge_inputs,export_inputs
from nfl_stats import code


def _toggle(config,key,label):
    config[key]=st.checkbox(label,value=config[key],key='nfl_layer_'+key)


def _number(config,key,label,step=.05,max_value=None):
    args=dict(min_value=0.,value=float(config[key]),step=step,key='nfl_layer_'+key)
    if max_value is not None:args['max_value']=float(max_value)
    config[key]=st.number_input(label,**args)


def controls_panel(data):
    config=data.setdefault('nfl_layer_settings',settings(data))
    st.subheader('NFL efficiency and game adjustments')
    _toggle(config,'enabled','Enable NFL add-on layers')
    st.caption('Active by default. The original scoring model and trained fit run first; these layers then adjust NFL projections and refresh market probabilities. The weights are editable scenario assumptions and have not been validated by the included reference fit.')
    with st.expander('Turn individual NFL layers on or off',expanded=True):
        columns=st.columns(2)
        options=[('efficiency_enabled','Passing/rushing EPA and smoothed success'),('dvoa_enabled','Opponent-adjusted DVOA baseline'),('trenches_enabled','Non-blitz pressure and line yards'),('weather_enabled','Sustained wind at 15+ MPH'),('rest_enabled','Rest difference and confirmed bye bonus'),('division_enabled','Division spread compression and lower total'),('underdog_enabled','Division underdog ranking preference')]
        for i,(key,label) in enumerate(options):
            with columns[i%2]:_toggle(config,key,label)
    with st.expander('Edit NFL layer weights'):
        st.caption('Passing EPA starts at 2.5× the weight of rushing EPA. Success rate smooths the efficiency signal; DVOA is strongest through week 8 and fades to its late-season weight by week 13.')
        a,b=st.columns(2)
        fields=[('pass_epa_weight','Passing EPA relative weight',.1,None),('rush_epa_weight','Rushing EPA relative weight',.1,None),('efficiency_blend','Additional efficiency blend',.05,1),('success_mix','Success share of additional efficiency',.05,1),('dvoa_early_weight','Early/mid-season DVOA weight',.05,1),('dvoa_late_weight','Late-season DVOA weight',.05,1),('rest_points_per_day','Margin points per extra rest day',.01,None),('bye_bonus','Extra margin bonus off confirmed bye',.1,None),('division_margin_shrink','Division margin shrink fraction',.01,1),('division_total_drop','Division total reduction (points)',.25,None),('division_dog_priority','Division underdog ranking bonus',.005,None),('short_home_dog_priority','Additional short home dog bonus',.005,None),('wind_total_at_threshold','Total reduction at 15 MPH',.25,None),('wind_total_per_mph','Extra total reduction per MPH',.05,None)]
        for i,(key,label,step,maximum) in enumerate(fields):
            with (a if i%2==0 else b):_number(config,key,label,step,maximum)
        st.caption('Ranking bonuses affect the order of positive-EV spread candidates. Quoted EV remains the probability/price calculation. A short home underdog means +0.5 through +3.5 by default. Manual weather points are replaced when a valid exposed-venue wind forecast triggers the automatic constraint.')
        if st.button('Reset only the NFL add-on weights',key='nfl_reset_layers'):
            data['nfl_layer_settings']=dict(DEFAULTS)
            for key in list(st.session_state):
                if key.startswith('nfl_layer_'):del st.session_state[key]
            rerun()
    with st.expander('NFL data coverage and optional provider inputs'):
        st.caption('A schedule/statistics refresh automatically builds player usage, actual active-game logs and a labeled xTD proxy from nflverse. Actual non-blitz pressure requires both pressure and blitz tags. Official DVOA, red-zone DVOA and adjusted line yards use dated provider imports. The PBP run-blocking fallback is labeled as an unadjusted line-yards proxy.')
        _toggle(config,'fetch_player_data','Load NFL player rosters and actual snaps on refresh')
        _toggle(config,'fetch_pressure_data','Try available NFL pressure/blitz charting on refresh')
        _toggle(config,'fetch_weather_data','Fetch sustained-wind forecasts for exposed NFL home venues')
        _toggle(config,'fetch_attd_odds','Fetch ATTD quotes on refresh when an Odds API key is supplied')
        st.caption('ATTD quotes use one extra event request per qualifying upcoming NFL game. Data coverage and your provider subscription determine availability. No requests run merely from changing weights.')
        st.write(f"Loaded: {len(data.get('nfl_team_inputs',[]))} dated team snapshots; {len(data.get('nfl_weather_inputs',[]))} wind forecasts; {len(data.get('nfl_player_candidates',[]))} players; {len(data.get('nfl_player_game_logs',[]))} player game logs; {len(data.get('nfl_attd_quotes',[]))} ATTD quotes.")
        for note in data.get('nfl_layer_feed_notes',[]):st.caption(note)
        st.caption('Optional uploads are in Setup & Saved Data → NFL provider inputs.')


def provider_inputs_panel(data):
    st.subheader('NFL provider inputs')
    uploaded=st.file_uploader('Import NFL advanced metrics, weather or player-prop inputs',type=['json'],key='nfl_layer_inputs_upload')
    if st.button('Apply NFL provider inputs',disabled=not uploaded,key='nfl_layer_inputs_apply'):
        try:
            payload=json.loads(uploaded.getvalue());updated=merge_inputs(data,payload)
            # Persist supplied player/depth/xTD/quote overrides across refresh.
            existing=merge_inputs({},data.get('nfl_provider_overrides',{'format':'nfl-layer-inputs-v1'}))
            from nfl_inputs import INPUT_KEYS
            existing=merge_inputs(existing,payload)
            updated['nfl_provider_overrides']=dict(format='nfl-layer-inputs-v1',**{k:existing.get(v,[]) for k,v in INPUT_KEYS.items()})
            data.clear();data.update(updated);rerun()
        except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
    example=Path(__file__).with_name('nfl_layer_inputs_example.json')
    if example.exists():st.download_button('Download NFL input schema/example',example.read_text(),file_name=example.name,mime='application/json',key='nfl_layer_example')
    st.download_button('Export NFL layers and player-prop inputs',export_inputs(data),file_name='nfl_layer_inputs.json',mime='application/json',key='nfl_layer_export')

def breakdown(result):
    extra=result.get('nfl_layers')
    if not extra:return
    st.write('NFL add-on adjustments')
    context=extra['context']
    st.caption(f"Division matchup: {context['divisional_matchup']}. Home-minus-away rest days: {context['rest_differential_home']}. Confirmed bye: away {context['away_off_bye']}, home {context['home_off_bye']}.")
    st.dataframe(pd.DataFrame(extra['terms']).T.rename_axis('Layer'),width='stretch')
    st.caption(f"Combined margin change: {extra['margin_adjustment']:+.2f}; combined total change: {extra['total_adjustment']:+.2f} points.")
    st.dataframe(pd.DataFrame(extra['team_features']).T,width='stretch')
    for note in extra['notes']:st.caption(note)


def attd_panel(data):
    st.divider();st.subheader('Anytime Touchdown Scorers')
    config=data.setdefault('nfl_layer_settings',settings(data))
    _toggle(config,'attd_enabled','Enable NFL anytime touchdown scorer')
    st.caption('Separate from ATS and total picks. The pool uses the quoted game total of at least 47.5 and high-volume RB1/2, WR1/2 and TE1 roles. Zero-TD and multiple-TD rules use the last active game, including zero-touch games with recorded snaps; byes and inactive games are skipped.')
    with st.expander('ATTD rules and weights'):
        for key,label in [('attd_recency_enabled','Bounce-back / sell-high recency rules'),('attd_redzone_enabled','Red-zone target and touch share'),('attd_xtd_enabled','Expected TDs versus actual TDs'),('attd_defense_enabled','Opponent defensive red-zone DVOA')]:_toggle(config,key,label)
        a,b=st.columns(2)
        for i,(key,label,step,maximum) in enumerate([('attd_min_total','Minimum Vegas game total',.5,None),('attd_bounce_bonus','Probability bonus after 0 TDs',.01,1),('attd_sell_multiplier','Probability multiplier after 2+ TDs',.01,1),('attd_min_rb_volume','Minimum RB carries + targets/game',1.,None),('attd_min_wr_volume','Minimum WR carries + targets/game',1.,None),('attd_min_te_volume','Minimum TE carries + targets/game',1.,None)]):
            with (a if i%2==0 else b):_number(config,key,label,step,maximum)
    if not config['enabled'] or not config['attd_enabled']:st.info('ATTD is switched off.');return
    games=[g for g in data['games'] if g.get('league')=='NFL']
    candidates=data.get('nfl_player_candidates',[]);logs=data.get('nfl_player_game_logs',[])
    if not candidates or not logs:
        st.info('Refresh the NFL slate to build player usage and active-game history, or import player-prop inputs. This frozen example has no player-prop history.');return
    opponents={}
    for game in games:
        for side in ('away','home'):opponents[code(game[side])]=eligible_team_metrics(data,game,game[side])
    as_of=datetime.now(timezone.utc).isoformat() if data.get('mode')=='connected' else forecast_as_of(data)
    automatic=[]
    for game in games:automatic.extend(eligible_reports(data,game))
    try:ranked=rate_anytime_touchdowns(candidates,logs,games,settings(data),as_of,opponents,data.get('nfl_attd_quotes',[]),data.get('injury_reports',[]),automatic,injury_settings(data))
    except (ValueError,TypeError,KeyError) as exc:st.error('ATTD inputs need attention: '+str(exc));return
    if ranked.empty:st.info('No players currently meet the game-total, role, volume and complete xTD-history filters.');return
    if ranked.opponent_redzone_dvoa.isna().any() and config['attd_defense_enabled']:st.caption('Opponent red-zone DVOA is missing for some players; those matchup terms are skipped.')
    if ranked.odds.isna().all():st.caption('No current matched ATTD prices: rankings show scenario probability only. Price-based value will appear when quotes are available.')
    display=ranked[['player','team','position','status','depth_rank','market_total','prior_active_week','prior_active_tds','opportunities_per_game','attd_probability','attd_if_active_probability','recency_rule','book','odds','conditional_ev']].copy()
    display['attd_probability']*=100;display['attd_if_active_probability']*=100;display['conditional_ev']*=100
    display=display.rename(columns={'player':'Player','team':'Team','position':'Position','status':'Availability','depth_rank':'Role rank','market_total':'Vegas total','prior_active_week':'Last active week','prior_active_tds':'Last active TDs','opportunities_per_game':'Carries + targets/game','attd_probability':'Scores in game scenario %','attd_if_active_probability':'If active scenario %','recency_rule':'Recency rule','book':'Book','odds':'ATTD odds','conditional_ev':'If active EV %'})
    st.dataframe(display,hide_index=True,width='stretch')
    st.caption('Current out/doubtful players are excluded. Questionable players receive an availability multiplier on the game scoring scenario. Price EV is conditional on being active; the export also shows availability-weighted EV assuming inactive bets are void. Confirm the book’s settlement rules and final active status.')
    st.caption('Probabilities include your requested adjustments and need calibration. Roles automatically use recent usage rank; imported depth ranks override them. Automatically built xTD is a PBP field-position frequency proxy. Actual snap coverage is required to activate automatic recency adjustments.')
    st.download_button('Download ATTD scorer details',ranked.to_csv(index=False),file_name='nfl_anytime_touchdown_scorers.csv',mime='text/csv',key='nfl_attd_export')
