"""Run with: python -m streamlit run app.py"""
from __future__ import annotations
import copy
import json
import os
from statistics import pstdev
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from workspace_runtime import rerun
from model import METRICS, CFB_METRICS, ensure_cfb_defaults, snapshot_season, parameters_for_league, quotes_for, validate
from data_pipeline import build_snapshot, normal_name, request_bytes
from cfb_metrics import apply_cfb_inputs, attach_fei, current_feed_warnings, preserve_cfb_overrides, roster_coverage
from calibration import archive, attach_results, candidate, signature
from control_help import CONTROL_HELP, METRIC_HELP, SCENARIO_HELP
from research_ui import historical_panel, injury_panel, clv_panel, clv_uploads, apply_injuries
from nfl_ui import historical_panel as nfl_historical_panel
from nfl_stats import NFL_METRICS
from cfb_preferences import apply_profile,flags as preference_flags,focus_signal
from trend_ui import panel as trend_panel
from nfl_layers import settings as nfl_settings
from auto_injuries import project,run,refresh as refresh_injuries,settings as injury_settings
from app_lifecycle import automatic_refresh,refresh_snapshot,preferences
from workspace_runtime import bootstrap,autosave,periodic_checks,startup_refresh,start_manual_refresh
from setup_ui import panel as setup_panel,injury_overview
from matchup_history import timestamp
from nfl_layers_ui import controls_panel as nfl_controls,attd_panel,breakdown as nfl_layer_breakdown,provider_inputs_panel

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Football Model Lab', page_icon='🏈', layout='wide')
# Let Streamlit theme every foreground and background together. Fixed pale
# backgrounds become unreadable when the browser selects Streamlit's dark theme.
st.markdown('''<style>
 [data-testid="stMetric"]{border:1px solid var(--border-color,#80808040);border-radius:10px;padding:15px}
 .block-container{padding-top:2rem}button{border-radius:7px!important}
 </style>''',unsafe_allow_html=True)


def reset_widgets():
    for key in list(st.session_state):
        if key.startswith(('weight_','scenario_','team_','selected_matchup_','nfl_layer_','auto_injury_','app_pref_')) or key in ('selected_matchup','dataset_json_editor'):del st.session_state[key]


def use_dataset(next_data):
    validate(next_data)
    ensure_cfb_defaults(next_data)
    next_data['nfl_layer_settings']=nfl_settings(next_data)
    next_data['auto_injury_settings']=injury_settings(next_data)
    next_data['app_preferences']=preferences(next_data)
    st.session_state.data=copy.deepcopy(next_data)
    st.session_state.baseline=copy.deepcopy(next_data)
    reset_widgets()


def secret(name):
    try:return st.secrets.get(name,os.getenv(name,''))
    except Exception:return os.getenv(name,'')


bootstrap(secret,ROOT,apply_profile)
data=st.session_state.data
# Network feeds run on a detached copy; a slow endpoint cannot hide Setup.
updated,changed=startup_refresh(data,secret)
if changed:use_dataset(updated);data=st.session_state.data

with st.sidebar:
    st.title('Football Model Lab')
    st.caption('NFL + CFB · Personal research workspace')
    st.subheader('Model controls')
    if st.button('Reset model parameters',use_container_width=True,key='reset_parameters'):
        data['parameters']=copy.deepcopy(st.session_state.baseline['parameters']);reset_widgets();rerun()
    p=data['parameters']
    selected_league=st.selectbox('League controls',['CFB','NFL'],format_func=lambda value:'College Football' if value=='CFB' else 'NFL',key='controls_league')
    lp=p[selected_league]
    st.caption('These controls affect '+('College Football' if selected_league=='CFB' else 'NFL')+' only.')
    for key,label,lo,hi,step in [('by','Yards per play weight',0.,12.,.1),('bs','Success rate weight',0.,1.5,.01),('shrink','Efficiency shrinkage',0.,1.,.01),('turnover_retention','Turnover retention',0.,1.,.01),('turnover_point_value','Turnover point value',0.,8.,.1)]:
        # Expand the widget range to preserve imported coefficients.
        value=float(lp.get(key,p[key]));lp[key]=st.slider(label,min(lo,value),max(hi,value),value,step=step,key='weight_'+selected_league+'_'+key,help=CONTROL_HELP[key])
        st.caption(CONTROL_HELP[key])
    with st.expander('Scoring, venue and uncertainty',expanded=True):
        for key,label,minimum in [('q','Baseline points per team',1.),('bx','Explosiveness weight',None),('h','Home-field margin',0.),('sigma_spread','Spread residual sigma',.1),('sigma_total','Total residual sigma',.1)]:
            kwargs=dict(value=float(lp[key]),step=.1,key='weight_'+selected_league+'_'+key)
            if minimum is not None:kwargs['min_value']=minimum
            lp[key]=st.number_input(label,help=CONTROL_HELP[key],**kwargs)
            st.caption(CONTROL_HELP[key])
    with st.expander('Reference metrics'):
        for key,label,minimum in [('n','Reference plays per team',1.),('y','Reference yards per play',0.),('s','Reference success %',0.),('x','Reference explosiveness',0.)]:
            lp[key]=st.number_input(label,min_value=minimum,value=float(lp[key]),step=.1,key='weight_'+selected_league+'_'+key,help=CONTROL_HELP[key])
    if selected_league == 'CFB':
        with st.expander('Your college-football selection profile',expanded=True):
            if st.button('Apply my offense and ranked-game profile',key='cfb_apply_preference_profile'):
                apply_profile(data);reset_widgets();rerun()
            lp['offense_share']=st.slider('Offense share of matchup efficiency',0.,1.,float(lp['offense_share']),step=.05,key='weight_CFB_offense_share',help=CONTROL_HELP['offense_share'])
            st.caption(CONTROL_HELP['offense_share'])
            lp['context_ranked_only']=st.checkbox('Use ranked opponents for letdown and look-ahead spots',value=lp['context_ranked_only'],key='weight_CFB_context_ranked_only')
            st.caption('Uses the latest eligible prior-week AP poll. Missing ranking history stays unknown. Turning this off uses the opponent-strength threshold below; manual rivalry flags work with either setting.')
            choices=['fade_spots','avoid_spots','standard']
            lp['selection_style']=st.selectbox('How these situations affect top picks',choices,index=choices.index(lp['selection_style']),format_func=lambda v:{'fade_spots':'Prioritize the opponent in these spots','avoid_spots':'Avoid games with these spots','standard':'Rank all eligible games by model EV'}[v],key='weight_CFB_selection_style')
            st.caption('Prioritize moves qualifying opponent spread picks ahead of other positive-EV picks. Avoid removes flagged games from top picks while keeping their projections visible. Score penalties still follow your context weights.')
            st.caption('Your starting profile: offense share 65%, full efficiency strength, FEI blend 40%, ranked letdown −2 margin points, ranked look-ahead −2, combined cap 3. These are your editable preferences, not historically fitted effects.')
        with st.expander('FEI and roster weights',expanded=True):
            lp['fei_weight']=st.slider('FEI forecast blend',0.,1.,float(lp['fei_weight']),step=.05,key='weight_CFB_fei_weight',help=CONTROL_HELP['fei_weight'])
            st.caption(CONTROL_HELP['fei_weight'])
            for key,label,minimum,step in [('fei_drives','Reference possessions per team',.1,.5),('starter_weight','Points per extra returning starter',0.,.05),('starters_reference','Returning starters reference per unit',0.,.5),('portal_weight','Portal points per normalized rating',0.,.1),('portal_scale','Portal native rating units per point scale',.1,.5),('roster_half_life','Roster weight half-life in games',.1,.5)]:
                args=dict(min_value=minimum,value=float(lp[key]),step=step,key='weight_CFB_'+key)
                if key=='starters_reference': args['max_value']=11.
                lp[key]=st.number_input(label,help=CONTROL_HELP[key],**args)
            st.caption('Roster weight = 2^(−games played / half-life). Portal ratings compare incoming minus outgoing talent on the same provider scale. Weights need historical calibration.')
    if selected_league=='NFL':
        with st.expander('NFL EPA and rest scenarios'):
            lp['epa_weight']=st.slider('Direct EPA forecast blend',0.,1.,float(lp['epa_weight']),step=.05,key='weight_NFL_epa_weight',help=CONTROL_HELP['epa_weight'])
            st.caption(CONTROL_HELP['epa_weight'])
            for key,label,minimum in [('epa_reference','Reference EPA per play',None),('rest_day_weight','Margin points per extra rest day',-2.),('rest_cap_points','Maximum direct rest-margin effect',0.)]:
                args=dict(value=float(lp[key]),step=.05,key='weight_NFL_'+key,help=CONTROL_HELP[key])
                if minimum is not None:args['min_value']=minimum
                if key=='rest_day_weight':args['max_value']=2.
                lp[key]=st.number_input(label,**args);st.caption(CONTROL_HELP[key])
    with st.expander('Opponent-adjusted game results',expanded=True):
        lp['power_weight']=st.slider('Prior result-rating blend',0.,1.,float(lp['power_weight']),step=.05,key='weight_'+selected_league+'_power_weight',help=CONTROL_HELP['power_weight'])
        st.caption(CONTROL_HELP['power_weight'])
        if selected_league=='CFB':
            method=st.selectbox('Point-rating method',['massey','ridge'],index=0 if lp['power_method']=='massey' else 1,format_func=lambda value:'Massey least squares' if value=='massey' else 'Regularized result ratings',key='weight_CFB_power_method')
            lp['power_method']=method
            st.caption('Massey predicts point strength from opponents and prior margins. Colley uses wins and losses and enters the trained gradient-boosting feature set; its rank is not added directly as score points.')
        lp['ml_weight']=st.slider('Validated gradient-boosting blend',0.,1.,float(lp['ml_weight']),step=.05,key='weight_'+selected_league+'_ml_weight',help=CONTROL_HELP['ml_weight'])
        st.caption(CONTROL_HELP['ml_weight'])
        lp['power_ridge']=st.number_input('Result-rating shrinkage',min_value=.01,value=float(lp['power_ridge']),step=.25,key='weight_'+selected_league+'_power_ridge',help=CONTROL_HELP['power_ridge'])
        lp['power_min_games']=st.number_input('Minimum prior games per team',min_value=1,value=int(lp['power_min_games']),step=1,key='weight_'+selected_league+'_power_min_games',help=CONTROL_HELP['power_min_games'])
        st.caption('This layer learns from actual prior margins and who each team played. Starting blend: 50% CFB, 25% NFL; these weights still need later-game validation.')
    with st.expander('Ten-year home and away ATS history'):
        for key,label,minimum in [('ats_weight','ATS record influence',0.),('ats_prior_games','ATS caution for sample size',.1),('ats_cap_points','Maximum ATS margin effect',0.)]:
            lp[key]=st.number_input(label,min_value=minimum,value=float(lp[key]),step=.1,key='weight_'+selected_league+'_'+key,help=CONTROL_HELP[key])
            st.caption(CONTROL_HELP[key])
    with st.expander('Head-to-head history'):
        for key,label in [('h2h_margin_weight','History weight on margin'),('h2h_total_weight','History weight on total')]:
            lp[key]=st.slider(label,0.,1.,float(lp[key]),step=.05,key='weight_'+selected_league+'_'+key,help=CONTROL_HELP[key])
            st.caption(CONTROL_HELP[key])
        for key,label in [('h2h_half_life_years','History half-life in years'),('h2h_prior_games','History prior sample size'),('h2h_difference_cap','Maximum historical discrepancy (points)')]:
            lp[key]=st.number_input(label,min_value=.1,value=float(lp[key]),step=.5,key='weight_'+selected_league+'_'+key,help=CONTROL_HELP[key])
        st.caption('Only completed meetings before the selected game count. Older games and small samples carry less weight.')
    if selected_league=='CFB':
        with st.expander('CFB look-ahead and previous-game context',expanded=True):
            for key,label in [('lookahead_margin_points','Look-ahead margin effect per team'),('big_win_margin_points','After a major win: margin effect per team'),('road_loss_home_margin_points','Road loss then home: margin boost'),('lookahead_total_points','Look-ahead total effect per team'),('big_win_total_points','After a major win: total effect per team')]:
                lp[key]=st.number_input(label,value=float(lp[key]),step=.25,key='weight_'+selected_league+'_'+key,help=CONTROL_HELP[key])
                st.caption(CONTROL_HELP[key])
            lp['context_window_days']=st.number_input('Maximum days between context games',min_value=1.,max_value=30.,value=float(lp['context_window_days']),step=1.,key='weight_'+selected_league+'_context_window_days',help=CONTROL_HELP['context_window_days'])
            lp['major_opponent_points']=st.number_input('Major opponent: strength above average (points)',min_value=0.,value=float(lp['major_opponent_points']),step=.5,key='weight_'+selected_league+'_major_opponent_points',help=CONTROL_HELP['major_opponent_points'])
            lp['context_cap_points']=st.number_input('Maximum combined context effect (points)',min_value=.01,value=float(lp['context_cap_points']),step=.25,key='weight_'+selected_league+'_context_cap_points',help=CONTROL_HELP['context_cap_points'])
            st.caption('Negative margin effects penalize the flagged team; positive effects favor it. Road loss then home starts at +0.5 margin points. Other effects start at zero. These are scenario assumptions; validate their effects on later games.')
    with st.expander('Margin and total calibration'):
        for key,label,minimum in [('margin_scale','Team-strength margin scale',.01),('margin_intercept','Margin intercept',None),('total_scale','Efficiency total scale',.01),('total_intercept','Total intercept',None)]:
            args=dict(value=float(lp[key]),step=.05,key='weight_'+selected_league+'_'+key)
            if minimum is not None:args['min_value']=minimum
            lp[key]=st.number_input(label,help=CONTROL_HELP[key],**args)
            st.caption(CONTROL_HELP[key])
        st.caption('Keep scales at 1 and intercepts at 0 until later-game validation supports a correction. Larger margin scale amplifies team-strength differences; home field, injuries and schedule-context points retain their units.')
    st.caption('All coefficients and residual sigmas start as uncalibrated priors. Changing them tests assumptions.')

st.caption('RESEARCH WORKSPACE / SLATE EXPLORER')
st.title('Explore the slate.')
st.write('Upcoming projections, your saved settings, and separate NFL touchdown scorers.')
periodic_checks()
if st.session_state.get('_storage_error'):st.error(st.session_state['_storage_error'])
if st.session_state.get('_refresh_error'):st.warning(st.session_state['_refresh_error'])
save_badge=st.empty()
mode=data.get('mode','snapshot')
st.info(('Frozen example. Open Setup & Saved Data for one-time keys, cloud connection and restore.' if mode=='snapshot' else 'Live workspace. Latest available inputs are reused; missing or old coverage is shown in Setup & Saved Data.')+' Conditional EV has not been validated.')
st.caption('As of '+str(data.get('as_of',data.get('snapshot_date','unspecified')))+' · '+data.get('window',''))
for warning in current_feed_warnings(data):st.warning(warning)
coverage=roster_coverage(data)
roster_gaps=[]
if data['parameters']['CFB']['starter_weight']>0 and coverage['missing_starters']:
    roster_gaps.append(f"{coverage['missing_starters']} lack complete usable returning-starter inputs")
if data['parameters']['CFB']['portal_weight']>0 and coverage['missing_portal']:
    roster_gaps.append(f"{coverage['missing_portal']} lack usable net portal ratings")
tabs=st.tabs(['College Football','NFL','Setup & Saved Data','Calibration & CLV','Method & validation'])
with tabs[1]:nfl_controls(data)
results=run(data)

def render_slate(league):
    if league=='NFL':st.subheader('ATS and Total Picks')
    if league=='CFB' and roster_gaps:
        st.warning(f"Optional CFB roster coverage ({coverage['slate_teams']} teams on this slate): "+'; '.join(roster_gaps)+'. Missing terms are skipped. See Setup & Saved Data → Data coverage and fixes.')
    league_results=[r for r in results if r['league']==league]
    ep=parameters_for_league(p,league)
    counts=st.columns(4)
    counts[0].metric('Games loaded',len(league_results));counts[1].metric('With projections',sum('away' in r for r in league_results));counts[2].metric('With usable odds',sum(bool(r['markets']) for r in league_results));counts[3].metric('Model version','Prototype 11')
    injury_overview(data,league)
    trend_panel(data,league)
    f=st.columns([1,1,2])
    market=f[0].selectbox('Market',['All','Spreads','Totals'],key='market_'+league)
    side_type=f[1].selectbox('Spread side',['All','Favorites','Underdogs'],disabled=market=='Totals',key='side_type_'+league)
    search=f[2].text_input('Search team or matchup',placeholder='Search '+league+' teams…',key='search_'+league)
    min_key='min_ev_'+league
    min_ev=st.slider('Minimum conditional EV %',0.,100.,step=1.,help=CONTROL_HELP['minimum_ev'],key=min_key,**({} if min_key in st.session_state else {'value':0.}))/100
    st.caption(CONTROL_HELP['minimum_ev'])
    st.caption('A spread signal is about beating a handicap. A +28 underdog projected to lose by 20 is a spread-cover candidate while the favorite remains the predicted game winner. These estimates require historical validation.')
    kind={'All':None,'Spreads':'spread','Totals':'total'}[market]
    def selection_score(q):return q.get('selection_score',q['conditional_ev']) if league=='NFL' else q['conditional_ev']
    def best(r):
        if league=='CFB' and p[league].get('selection_style')=='avoid_spots' and any(preference_flags(r,s) for s in ('away','home')):return None
        def eligible(q):
            if data.get('mode')=='connected':
                try:
                    now=datetime.now(timezone.utc);g=r['game'];at=timestamp(q.get('updated_at'))
                    if g.get('completed') or timestamp(g.get('kickoff'))<=now or not timedelta(0)<=now-at<=timedelta(hours=6) or not q.get('book') or q['book']=='Manual scenario':return False
                except (ValueError,TypeError):return False
            if q['conditional_ev']<=min_ev or (kind is not None and q['kind']!=kind):return False
            if side_type!='All' and market!='Totals':return q['kind']=='spread' and (q['line']<0 if side_type=='Favorites' else q['line']>0)
            return True
        return max((q for q in r['markets'] if eligible(q)),key=lambda q:(int(focus_signal(r,q)) if league=='CFB' and p[league].get('selection_style')=='fade_spots' else 0,selection_score(q)),default=None)
    def names(g):return data['teams'].get(g['away'],{}).get('name',g['away'])+' @ '+data['teams'].get(g['home'],{}).get('name',g['home'])
    visible=[r for r in results if (league=='All' or r['league']==league) and (not search or search.lower() in (names(r['game'])+' '+r['game']['away']+' '+r['game']['home']).lower())]
    st.subheader('Largest model discrepancies')
    style=p[league].get('selection_style','standard') if league=='CFB' else 'standard'
    st.caption('Top two '+league+' games under your selection controls; one quoted market per game. '+('Qualifying opponent spreads in letdown/look-ahead spots receive priority.' if style=='fade_spots' else 'Ranked by conditional EV plus enabled divisional underdog preference.' if league=='NFL' else 'Ranked by conditional EV.')+' These are model estimates.')
    for lg in [league]:
        ranked=sorted([r for r in visible if r['league']==lg and best(r)],key=lambda r:(int(focus_signal(r,best(r))) if style=='fade_spots' else 0,selection_score(best(r))),reverse=True)[:2]
        st.markdown('**'+lg+'**')
        if not ranked:st.caption('No games with usable odds match this filter.')
        for col,r in zip(st.columns(2),ranked):
            q=best(r)
            with col:
                with st.container(border=True):
                    st.markdown('**'+names(r['game'])+'**')
                    st.metric('Projected score · Away – Home',f"{r['away']['score']:.1f} – {r['home']['score']:.1f}")
                    st.write(f"{q['label']} · {q['odds']:+g} · {q['book']}")
                    st.caption(f"{q['edge']:+.2f} points discrepancy · conditional EV {100*q['conditional_ev']:+.1f}%")
                    if league=='CFB' and focus_signal(r,q):st.caption('Matches your ranked-opponent situation focus.')
                    if league=='NFL' and q.get('ats_priority_bonus'):st.caption(f"Divisional underdog ranking preference: +{100*q['ats_priority_bonus']:.1f} points in the ranking score.")
    st.subheader('Every loaded game')
    table=[]
    for r in visible:
        g=r['game'];q=best(r);row={'Matchup':names(g),'League':r['league'],'Status':r['status']}
        if 'away' in r:row.update({'Away score':r['away']['score'],'Home score':r['home']['score'],'Reference away spread':g.get('line_away'),'Model away spread':r['fair_line_away'],'Away spread difference':r['spread_edge_away'],'Reference total':g.get('total'),'Model total':r['fair_total'],'Over difference':r['total_edge_over']})
        if 'away' in r:
            row['Projected winner']=data['teams'][g['away'] if r['margin_away']>0 else g['home']].get('name') if r['margin_away']!=0 else 'Tie'
            row['Result-rating margin effect']=r['power_rating']['margin_adjustment'];row['ATS margin effect']=r['ats_history']['margin_adjustment']
            row['Trained model active']='Active' if r.get('gradient_boosting',{}).get('active',False) else 'Inactive'
            if league=='CFB':
                row['Away situational flags']=', '.join(preference_flags(r,'away'))
                row['Home situational flags']=', '.join(preference_flags(r,'home'))
            if league=='NFL' and r.get('nfl_layers'):
                extra=r['nfl_layers'];row['Divisional matchup']=extra['context']['divisional_matchup']
                row['NFL add-on margin effect']=extra['margin_adjustment'];row['NFL add-on total effect']=extra['total_adjustment']
            for side in ('away','home'):
                records=data['teams'].get(g[side],{}).get('ats_records',{})
                row[side.title()+' team Home ATS']=records.get('home',{}).get('record')
                row[side.title()+' team Away ATS']=records.get('away',{}).get('record')
        row.update({'Best quoted market':q['label'] if q else None,'Book':q['book'] if q else None,'Conditional EV %':100*q['conditional_ev'] if q else None})
        extras=r.get('cfb_extensions')
        if extras:
            row.update({'FEI used':extras['fei_active'] and extras['fei_weight']>0,'Portal used':extras['portal_active'] and extras['parameters']['portal_weight']>0,'Away roster decay':extras['roster_decay']['away'],'Home roster decay':extras['roster_decay']['home']})
        if r.get('head_to_head'):row['Prior meetings']=r['head_to_head']['count']
        context=r.get('schedule_context')
        if context:
            row.update({'Away look-ahead':context['away']['lookahead'],'Home look-ahead':context['home']['lookahead'],'Away after major win':context['away']['big_win'],'Home after major win':context['home']['big_win'],'Home after road loss':context['home']['road_loss_return_home'],'Schedule margin effect':context['margin_adjustment'],'Schedule total effect':context['total_adjustment']})
        table.append(row)
    df=pd.DataFrame(table)
    if not df.empty:
        if 'Conditional EV %' in df:df=df.sort_values('Conditional EV %',ascending=False,na_position='last')
        st.dataframe(df,width='stretch',hide_index=True,height=min(650,38*len(df)+40))
        st.download_button('Download displayed projections',df.to_csv(index=False),file_name=league.lower()+'_projections.csv',mime='text/csv',key='projections_download_'+league)
    else:st.caption('No games match your filters.')
    st.caption('Reference lines may be multi-book medians. Conditional EV uses each quoted line and price. Live top picks require an upcoming game and a named-book quote observed within six hours; old or manual quotes remain inspectable in the breakdown.')
    if visible:
        game_ids=[results.index(r) for r in visible]
        index=st.selectbox('Inspect a matchup',game_ids,format_func=lambda i:names(results[i]['game']),key='selected_matchup_'+league)
        r=results[index];g=data['games'][index]
        with st.expander('Projection breakdown',expanded=True):
            if 'away' in r:
                breakdown=pd.DataFrame({g['away']:r['away']['score_parts'],g['home']:r['home']['score_parts']})
                st.dataframe(breakdown,width='stretch')
                st.write('Matched efficiency inputs')
                st.dataframe(pd.DataFrame({g['away']:r['away']['matched'],g['home']:r['home']['matched']}),width='stretch')
                st.write(f"Pace factor = {r['pace']:.6f}. Fair away spread = {r['fair_line_away']:+.6f}. Fair total = {r['fair_total']:.6f}.")
                for side in ('away','home'):
                    m=r[side]['matched'];lg=data['parameters'][r['league']]
                    st.code(f"{g[side]}: B = {lg['q']} + {ep['shrink']} × [{ep['by']}×({m['y']:.4f}−{lg['y']}) + {ep['bs']}×({m['s']:.4f}−{lg['s']}) + {lg['bx']}×({m['x']:.4f}−{lg['x']})] = {r[side]['base']:.6f}\nScore = "+' '.join(f'{v:+.6f}' for v in r[side]['score_parts'].values())+f" = {r[side]['score']:.6f}")
                st.dataframe(pd.DataFrame(r['markets']),width='stretch',hide_index=True)
                power=r['power_rating']
                st.write('Opponent-adjusted prior results')
                st.caption(power['status'])
                if power['active']:
                    st.write(f"Away neutral rating: {power['away_rating']:+.2f}; home neutral rating: {power['home_rating']:+.2f}. Their predicted neutral margin is {power['neutral_margin']:+.2f}, using {power['prior_games']} prior games across the league.")
                    st.write(f"Result-rating blend: {power['weight']:.0%}; margin change: {power['margin_adjustment']:+.2f} points. Home field and the other explicit modifiers are applied separately.")
                if league=='CFB':
                    st.write('Colley and Massey inputs')
                    st.dataframe(pd.DataFrame([{'Team':g[side],'Colley rating':power.get('colley_'+side),'Massey point rating':power.get('massey_'+side)} for side in ('away','home')]),hide_index=True,width='stretch')
                if league=='NFL':
                    st.write('NFL advanced efficiency and rest')
                    st.dataframe(pd.DataFrame([{'Team':data['teams'][g[s]].get('name',g[s]),**{key:data['teams'][g[s]].get(key) for key in NFL_METRICS}} for s in ('away','home')]),hide_index=True,width='stretch')
                    extra=r.get('nfl_extensions',{});st.caption(extra.get('status',''))
                    st.write(f"Rest days: away {extra.get('rest_days',{}).get('away')}, home {extra.get('rest_days',{}).get('home')}. Direct home-rest margin effect {extra.get('rest_margin_home',0.):+.2f}.")
                    nfl_layer_breakdown(r)
                injury=r.get('automatic_injuries',{})
                if injury.get('rows'):
                    st.caption(f"Automatic injury score change: away {injury['away']:+.2f}, home {injury['home']:+.2f}.")
                    st.dataframe(pd.DataFrame(injury['rows']),hide_index=True,width='stretch')
                boost=r.get('gradient_boosting',{})
                st.caption(boost.get('status','No trained model loaded.'))
                st.write(f"Gradient-boosting margin correction: {boost.get('margin_adjustment',0.):+.2f}; total correction: {boost.get('total_adjustment',0.):+.2f} points.")
                injury=g.get('injury_import')
                if injury:
                    st.caption('Reviewed injury coverage: '+json.dumps(injury['coverage']))
                st.write('Ten-year home and away ATS records')
                ats_rows=[]
                for side in ('away','home'):
                    team=data['teams'].get(g[side],{});history=team.get('ats_records',{})
                    for venue in ('home','away'):
                        record=history.get(venue,{})
                        ats_rows.append({'Team':team.get('name',g[side]),'Venue':venue.title(),'ATS W-L-P':record.get('record'),'Cover %':100*record['cover_rate'] if record.get('cover_rate') is not None else None,'Missing lines':record.get('missing_lines'),'Unknown venues':record.get('unknown_venues'),'Window start':history.get('window_start'),'Through':history.get('as_of')})
                    if history:st.caption(team.get('name',g[side])+': '+history.get('source','')+('; missing seasons '+str(history['missing_years']) if history.get('missing_years') else ''))
                st.dataframe(pd.DataFrame(ats_rows),hide_index=True,width='stretch')
                st.caption(r['ats_history']['status'])
                st.write(f"ATS history margin change: {r['ats_history']['margin_adjustment']:+.3f} points. Neutral sites are excluded from home/away records. Pushes do not count toward the cover percentage.")
                extras=r.get('cfb_extensions')
                if extras:
                    st.write('FEI and roster contribution')
                    st.write(f"FEI blend: {extras['fei_weight']:.0%}. Estimated possessions per team: {extras['expected_drives']:.2f}.")
                    if extras['fei_active']:
                        st.code('\n'.join(f"{g[side]}: FEI forecast = {r[side]['fei_score']:.6f}; blend correction = {r[side]['score_parts']['fei_blend']:+.6f}; returning starters = {r[side]['score_parts']['returning_starters']:+.6f}; portal = {r[side]['score_parts']['portal']:+.6f}" for side in ('away','home')))
                    for note in dict.fromkeys(extras['notes']): st.caption(note)
                history=r.get('head_to_head',{})
                st.write('Head-to-head history')
                if history.get('count'):
                    st.write(f"{history['count']} previous meetings; current away team won {history['wins_away']}. Age-weighted sample: {history['weighted_games']:.2f}; shrinkage: {history['shrinkage']:.3f}.")
                    st.write(f"History margin adjustment: {history['margin_adjustment']:+.3f}; total adjustment: {history['total_adjustment']:+.3f} points.")
                    st.dataframe(pd.DataFrame(history['rows']),width='stretch',hide_index=True)
                else:st.caption(history.get('status','No history loaded. Refresh the slate to load historical meetings.'))
                context=r['schedule_context'];context_rows=[]
                st.write('Look-ahead and previous-game context')
                for side in ('away','home'):
                    info=context[side];previous=info['previous'] or {};following=info['next'] or {}
                    context_rows.append({'Team':data['teams'][g[side]].get('name',g[side]),'Next opponent':following.get('opponent'),'Days to next game':following.get('days_until'),'Next opponent strength':following.get('opponent_strength'),'Previous opponent':previous.get('opponent'),'Previous venue':previous.get('venue'),'Previous margin':previous.get('margin'),'Days since previous game':previous.get('days_since'),'Look-ahead':info['lookahead'],'Major win':info['big_win'],'Road loss then home':info['road_loss_return_home']})
                    for signal,label in [('lookahead','Look-ahead'),('big_win','Major win'),('road_loss_return_home','Road loss then home')]:
                        flag='Unknown' if info[signal] is None else 'Yes' if info[signal] else 'No'
                        st.caption(f"{g[side]} · {label}: {flag} · {info['sources'][signal]} · {info['reasons'].get(signal,'Does not qualify')}")
                st.dataframe(pd.DataFrame(context_rows),width='stretch',hide_index=True)
                dm,dt=context['margin_adjustment'],context['total_adjustment']
                st.code('Margin terms: '+json.dumps(context['margin_terms'])+'\nTotal terms: '+json.dumps(context['total_terms'])+f"\nCapped margin effect = {dm:+.3f}; total effect = {dt:+.3f}\nAway score effect = ({dm:+.3f} + {dt:+.3f}) / 2 = {(dm+dt)/2:+.3f}\nHome score effect = ({dt:+.3f} − {dm:+.3f}) / 2 = {(dt-dm)/2:+.3f}")
                if g.get('context_notes'):st.caption('Context note: '+g['context_notes'])
            else:st.warning(r['status']+' '+', '.join(r.get('missing',[])))
            st.caption(g.get('notes',''))
            if g.get('modifiers_verified') is False:st.warning('Injury, weather and rest modifiers have not been verified for this pull. Defaults are zero.')
        with st.expander('Edit matchup assumptions'):
            with st.form('scenario_form_'+str(index)):
                cols=st.columns(3);values={}
                defaults={'line_away':0.,'total':45.,'odds_away':-110.,'odds_home':-110.,'odds_over':-110.,'odds_under':-110.,'injury_away':0.,'injury_home':0.,'rest_margin_home':0.,'weather_total':0.}
                labels={'line_away':'Away spread','total':'Total','odds_away':'Away odds','odds_home':'Home odds','odds_over':'Over odds','odds_under':'Under odds','injury_away':'Away injury points','injury_home':'Home injury points','rest_margin_home':'Rest margin to home','weather_total':'Weather total adjustment'}
                qs=quotes_for(g)
                for i,(key,fallback) in enumerate(defaults.items()):
                    value=g.get(key)
                    if value is None and key.startswith('odds_'):value=next((q['odds'] for q in qs if q['selection']==key[5:]),fallback)
                    value=float(fallback if value is None else value)
                    step=1. if key.startswith('odds_') else .5 if key in ('line_away','total') else .1
                    explanation=SCENARIO_HELP.get(key,'American odds: −110 means risking 110 units for 100 units of net profit; +150 means 150 units of net profit per 100 risked. This changes price-based EV, not the score.')
                    values[key]=cols[i%3].number_input(labels[key],value=value,step=step,key='scenario_'+str(index)+'_'+key,help=explanation)
                values['neutral']=st.checkbox('Neutral venue',value=g.get('neutral',False),key='scenario_'+str(index)+'_neutral')
                if st.form_submit_button('Apply scenario'):
                    modified={**g,**values}
                    if any(values[k]!=g.get(k,0.) for k in ('injury_away','injury_home')):modified.pop('injury_import',None)
                    quote_changed=any(values[k]!=(g.get(k) if g.get(k) is not None else next((q['odds'] for q in qs if q['selection']==k[5:]),defaults[k]) if k.startswith('odds_') else defaults[k]) for k in ('line_away','total','odds_away','odds_home','odds_over','odds_under'))
                    if quote_changed or not qs:
                        modified.pop('quotes',None);modified['book']='Manual scenario';modified.pop('odds_updated_at',None)
                    try:
                        project(data,modified)
                        data['games'][index]=modified;rerun()
                    except (ValueError,ArithmeticError) as exc:st.error(str(exc))
        with st.expander('Edit schedule context'):
            st.caption('Automatic uses loaded schedules and frozen opponent strength. On/Off creates a manual scenario, which is retained in your JSON export. Refresh the slate to load schedules for this feature.')
            with st.form('context_form_'+str(index)):
                context_values={};columns=st.columns(2);options=['Automatic','On','Off']
                for col,side in zip(columns,('away','home')):
                    context_values[side]={};col.markdown('**'+g[side]+'**')
                    fields=[('lookahead','Look-ahead flag'),('big_win','Previous major-win flag')]
                    if side=='home' and not g.get('neutral'):fields.append(('road_loss_return_home','Road loss then home flag'))
                    for signal,label in fields:
                        current=g.get('context_overrides',{}).get(side,{}).get(signal)
                        choice=col.selectbox(g[side]+' · '+label,options,index=0 if current is None else 1 if current else 2,key='scenario_context_'+str(index)+'_'+side+'_'+signal)
                        context_values[side][signal]=None if choice=='Automatic' else choice=='On'
                context_note=st.text_input('Context note (rivalry, important opponent, source)',value=g.get('context_notes',''),key='scenario_context_'+str(index)+'_note')
                if st.form_submit_button('Apply schedule context'):
                    modified={**g,'context_overrides':context_values,'context_notes':context_note}
                    try:
                        validate({**data,'games':[modified]});project(data,modified)
                        data['games'][index]=modified;rerun()
                    except ValueError as exc:st.error(str(exc))
        with st.expander('Edit team metrics'):
            available=[tid for tid in (g['away'],g['home']) if tid in data['teams']]
            if available:
                team_id=st.selectbox('Team',available,key='team_id_'+league)
                with st.form('team_form_'+team_id):
                    current=data['teams'][team_id];inputs={};cols=st.columns(4)
                    for i,key in enumerate(METRICS):inputs[key]=cols[i%4].number_input(key,value=float(current.get(key) or 0),step=.1,key='team_'+team_id+'_'+key,help=METRIC_HELP[key])
                    st.caption('y/dy: offense/allowed YPP; s/ds: success %; x/dx: explosive metric; p: plays per game; t: net turnovers per game. Zero fields above require your actual input before applying.')
                    if st.form_submit_button('Apply team statistics'):
                        data['teams'][team_id].update(inputs);rerun()
        if r['league']=='CFB':
            with st.expander('Edit FEI and roster inputs'):
                available=[tid for tid in (g['away'],g['home']) if tid in data['teams']]
                if available:
                    tid=st.selectbox('CFB team',available,format_func=lambda tid:data['teams'][tid].get('name',tid),key='team_cfb_id')
                    current=data['teams'][tid]
                    with st.form('cfb_inputs_'+tid):
                        columns=st.columns(2); inputs={}
                        labels={'ofei':'Offensive FEI (OFEI)','dfei':'Defensive FEI (DFEI)','returning_starters_off':'Returning offensive starters (0–11)','returning_starters_def':'Returning defensive starters (0–11)','portal_net_rating':'Net portal rating (sum in minus out)','roster_games_played':'Games played for roster decay'}
                        for i,key in enumerate(CFB_METRICS):
                            value=current.get(key)
                            if key=='roster_games_played' and value is None: value=current.get('sample_games')
                            args=dict(value=value,key='team_cfb_'+tid+'_'+key)
                            if key in ('returning_starters_off','returning_starters_def','roster_games_played'):
                                args.update(min_value=0,step=1,value=int(value) if value is not None else None)
                                if key.startswith('returning'): args['max_value']=11
                            else: args.update(step=.01,value=float(value) if value is not None else None)
                            inputs[key]=columns[i%2].number_input(labels[key],**args)
                        year=snapshot_season(data) or datetime.now().year
                        fei_season=columns[0].number_input('FEI season',value=int(current.get('fei_season') or year),step=1,key='team_cfb_'+tid+'_fei_season')
                        through=current.get('fei_through_week')
                        fei_week=columns[1].number_input('FEI through week',min_value=0,value=int(through) if through is not None else None,step=1,key='team_cfb_'+tid+'_fei_week')
                        roster_season=columns[0].number_input('Roster season',value=int(current.get('roster_season') or year),step=1,key='team_cfb_'+tid+'_roster_season')
                        st.caption('Blank means unknown. Positive DFEI is a stronger defense. Use the numeric FEI rating, not its rank. Portal sums must use one consistent provider rating scale.')
                        if st.form_submit_button('Apply FEI and roster inputs'):
                            record={k:v for k,v in inputs.items() if v!=current.get(k)}
                            if any(k in record for k in ('ofei','dfei')) or fei_season!=current.get('fei_season') or fei_week!=current.get('fei_through_week'):
                                record.update(fei_season=fei_season,fei_through_week=fei_week,fei_source='Manual FEI inputs')
                            if any(k in record for k in ('returning_starters_off','returning_starters_def','portal_net_rating','roster_games_played')) or roster_season!=current.get('roster_season'):
                                record.update(roster_season=roster_season,roster_source='Manual roster inputs')
                            try:
                                updated,_,_=apply_cfb_inputs(data,{tid:record});use_dataset(updated);rerun()
                            except ValueError as exc: st.error(str(exc))

with tabs[2]:
    setup_status=setup_panel(data,secret,reset_widgets,results)
    with st.expander('CFB historical data, training and trained-model restore'):
        historical_panel(data,secret,reset_widgets)
    with st.expander('NFL historical data, training and trained-model restore'):
        nfl_historical_panel(data,reset_widgets)
    with st.expander('NFL advanced metrics, weather and player-prop uploads'):
        provider_inputs_panel(data)
    injury_panel(data,'CFB')
    injury_panel(data,'NFL')
    with st.expander('Optional CLV ledgers and closing-quote uploads'):
        clv_uploads(data,'CFB');clv_uploads(data,'NFL')
    with st.expander('Optional calibration archive upload'):
        uploaded=st.file_uploader('Import calibration archive JSON',type=['json'],key='calibration_upload')
        if st.button('Load calibration archive',disabled=not uploaded,key='calibration_load'):
            try:
                payload=json.loads(uploaded.getvalue())
                if payload.get('format')!='pregame-calibration-v1' or not isinstance(payload.get('forecasts'),list):raise ValueError('Use a pregame calibration archive.')
                st.session_state['calibration_archive']=payload;rerun()
            except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
    st.subheader('Data coverage and fixes')
    st.write('Yellow messages identify data gaps. Missing optional roster terms are skipped; complete core statistics can still produce scores. Games without verified market quotes have no market EV.')
    odds_diagnostics=data.get('odds_join_diagnostics',{})
    if odds_diagnostics:
        st.dataframe(pd.DataFrame([{'League':lg,'Odds events received':d.get('received_events',0),'Outside selected dates':d.get('outside_window',0),'Matched events':d.get('matched_events',0),'Unmatched events':len(d.get('unmatched_events',[])),'Matched with no usable quotes':d.get('matched_without_quotes',0)} for lg,d in odds_diagnostics.items()]),hide_index=True,width='stretch')
        failed=[{'league':lg,**event} for lg,d in odds_diagnostics.items() for event in d.get('unmatched_events',[])]
        if failed:
            st.dataframe(pd.DataFrame(failed),hide_index=True,width='stretch')
            st.write('For an unmapped name, confirm which schedule team it represents and save a mapping below. For a date mismatch, refresh the schedule and check provider kickoff dates. Duplicate schedule matches require correcting the input schedule. CFB coverage is limited to games involving FBS teams.')
    elif data.get('mode')!='snapshot':
        st.info('Refresh the slate with this version to load event-by-event odds diagnostics.')
    st.caption('The odds feed returns bookmaker-listed events. A schedule can include games whose lines are not posted yet, and completed games have no current odds. An empty market is not automatically a matching error.')
    if coverage['rows']:
        st.dataframe(pd.DataFrame(coverage['rows']),hide_index=True,width='stretch')
        st.write('Returning starters: select the matchup in College Football → Edit FEI and roster inputs. Enter verified offensive and defensive counts (0–11), the roster season and games played. Bulk entry is available in the CFB input template below.')
        st.write('Portal ratings: the table shows missing dates, player ratings and eligibility. Refresh to check for provider corrections, or enter a verified full incoming-minus-outgoing sum using one provider scale. A partial sum or a made-up zero does not replace missing data.')
        st.caption('These counts include only teams in the selected slate and update after edits. Setting a roster weight to 0 disables that term; it does not complete its missing inputs.')
    with st.expander('Verified odds team-name mappings'):
        st.write('Use the exact provider team name from the diagnostics table and select its verified schedule team. Saved mappings apply on the next refresh and remain in your exported snapshot.')
        existing=[{'League':lg,'Provider name':name,'Team ID':tid} for lg,values in data.get('odds_alias_overrides',{}).items() for name,tid in values.items()]
        if existing:st.dataframe(pd.DataFrame(existing),hide_index=True,width='stretch')
        mapping_leagues=[lg for lg in ('NFL','CFB') if any(t.get('league')==lg for t in data['teams'].values())]
        if mapping_leagues:
            mapping_league=st.selectbox('Mapping league',mapping_leagues,key='odds_alias_league')
            mapping_teams=[tid for tid,t in data['teams'].items() if t.get('league')==mapping_league]
            with st.form('odds_alias_form'):
                provider_name=st.text_input('Exact odds-provider team name',key='odds_alias_name')
                target=st.selectbox('Verified schedule team',mapping_teams,index=None,format_func=lambda tid:data['teams'][tid].get('name',tid)+' ('+tid+')',placeholder='Select a verified team',key='team_odds_alias_target')
                save_mapping=st.form_submit_button('Save verified mapping')
                remove_mapping=st.form_submit_button('Remove mapping for this provider name')
                if save_mapping or remove_mapping:
                    name=provider_name.strip()
                    if not normal_name(name):st.error('Enter the exact provider team name.')
                    elif save_mapping and target is None:st.error('Select the verified schedule team.')
                    else:
                        saved=data.setdefault('odds_alias_overrides',{}).setdefault(mapping_league,{})
                        for old_name in list(saved):
                            if normal_name(old_name)==normal_name(name):saved.pop(old_name)
                        if save_mapping:saved[name]=target
                        st.success('Mapping saved.' if save_mapping else 'Mapping removed.')
                        st.info('Refresh schedules, statistics and odds below to apply this change.')
    diagnostic_report=dict(as_of=data.get('as_of'),window=data.get('window'),warnings=current_feed_warnings(data),cfb_roster_coverage=coverage,odds_join_diagnostics=odds_diagnostics,odds_alias_overrides=data.get('odds_alias_overrides',{}))
    st.download_button('Download data diagnostics',json.dumps(diagnostic_report,indent=2,allow_nan=False),file_name='football_data_diagnostics.json',mime='application/json')
    st.subheader('Load a full slate')
    st.write('Automatic loading uses upcoming games over the next seven days and your saved host keys. Use this form for a different date range or a refresh now.')
    today=datetime.now(ZoneInfo('America/New_York')).date()
    with st.form('data_refresh'):
        c=st.columns(3)
        season=c[0].number_input('Season',min_value=1999,max_value=2100,value=today.year,step=1)
        start=c[1].date_input('First date',today)
        end=c[2].date_input('Last date',today+timedelta(days=7))
        leagues=st.multiselect('Leagues',['NFL','CFB'],default=['NFL','CFB'])
        c=st.columns(2)
        cfbd_key=c[0].text_input('CollegeFootballData API key',value=secret('CFBD_API_KEY'),type='password',key='cfb_api_live')
        odds_key=c[1].text_input('The Odds API key',value=secret('ODDS_API_KEY'),type='password',key='odds_api_live')
        regions=st.selectbox('Book region',['us','us2','uk','eu','au'])
        refresh=st.form_submit_button('Refresh schedules, statistics and odds')
    if refresh:
        start_manual_refresh(data,int(season),start,end,leagues,cfbd_key,odds_key,regions)
        rerun()
    st.caption('NFL statistics use eligible prior-week plays. CFB uses prior-week advanced stats plus box scores. Feed access, quota and publication timing determine coverage. First-week teams need preseason priors or manually supplied statistics.')
    st.caption('Refresh also loads current-season schedules for next opponents and previous results. Missing schedules or opponent strength remain unknown. The frozen example needs a schedule refresh for automatic context flags.')
    st.caption('Refresh fits opponent-adjusted ratings from prior completed results and loads ten-year home/away ATS history. The first CFB history load requests past schedules and lines; later refreshes reuse the private cloud cache when connected. API access and missing records determine coverage.')
    with st.expander('Download historical data for training'):
        st.markdown('**NFL:** [nflverse play-by-play CSV downloads](https://github.com/nflverse/nflverse-data/releases/tag/pbp) and [schedule, results and closing spreads CSV](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv). **CFB:** [CollegeFootballData CSV exporter](https://collegefootballdata.com/exporter) and [API reference](https://api.collegefootballdata.com/).')
        st.markdown('NFL downloads are public and need no API key. CFB offers a [free key](https://collegefootballdata.com/key) with 1,000 calls/month, or a free academic plan with 3,000 calls/month when registering with a `.edu` email. Historical games, team stats, advanced metrics and betting lines are included; check the [current access limits](https://collegefootballdata.com/api-tiers). Cache past seasons to conserve requests.')
        st.write('Start with 2016–2025. Export game results, game/weekly statistics, advanced statistics and historical lines. Create one training row per game from earlier completed games only; attach that game’s final scores as targets. Keep NFL and CFB separate.')
        st.write('For a clean chronological split, fit on 2016–2022, choose settings on 2023–2024, and test once on untouched 2025. Repeat with rolling seasons to check stability. Do not use end-of-season statistics to predict earlier games.')
        st.write('Open the CFB history section here for game-level CFBD files or API downloads. Open the NFL history section here to download free nflverse seasons or upload games.csv and play-by-play files. Both sections reconstruct prior-week features, train independent gradient-boosting models, and evaluate later seasons.')
        st.download_button('Download pregame training-row template',(ROOT/'training_template.json').read_text(),file_name='pregame_training_template.json',mime='application/json')
        st.caption('Read training_guide.md in the deployment package for download steps, timestamps, ATS line definitions and evaluation metrics.')
    st.subheader('CFB FEI and roster inputs')
    st.write('FEI loads from BCF Toys during a CFB refresh. Portal ratings load through your CFBD key. Returning starters need verified offensive and defensive counts. Manual overrides are retained in exported JSON and reapplied within the same season on a slate refresh.')
    cfb_teams={tid:t for tid,t in data['teams'].items() if t.get('league')=='CFB'}
    if cfb_teams:
        st.dataframe(pd.DataFrame([{'Team':t.get('name',tid),'OFEI':t.get('ofei'),'DFEI':t.get('dfei'),'FEI through week':t.get('fei_through_week'),'Returning offense':t.get('returning_starters_off'),'Returning defense':t.get('returning_starters_def'),'Net portal rating':t.get('portal_net_rating'),'Games played':t.get('roster_games_played') if t.get('roster_games_played') is not None else t.get('sample_games')} for tid,t in cfb_teams.items()]),hide_index=True,width='stretch')
        if st.button('Load latest FEI for loaded CFB teams'):
            try:
                updated=copy.deepcopy(data); year=snapshot_season(data) or int(season)
                cutoffs=[g['feature_through_week'] for g in data['games'] if g.get('league')=='CFB' and g.get('feature_through_week') is not None]
                attach_fei(updated['teams'],request_bytes(f'https://bcftoys.com/{year}-fei',ttl=3600).decode('utf-8'),year,min(cutoffs) if cutoffs else None)
                for record in updated.get('cfb_overrides',{}).values():
                    for key in ('ofei','dfei','fei_season','fei_through_week','fei_source'): record.pop(key,None)
                use_dataset(updated);rerun()
            except (RuntimeError,ValueError,UnicodeError) as exc: st.error(str(exc))
        template={'teams':{t.get('name',tid):{**{k:None for k in CFB_METRICS},'fei_season':snapshot_season(data),'fei_through_week':None,'roster_season':snapshot_season(data)} for tid,t in cfb_teams.items()}}
        st.download_button('Download CFB input template',json.dumps(template,indent=2),file_name='cfb_inputs.json',mime='application/json')
        supplement=st.file_uploader('Import FEI and roster JSON',type=['json'],key='cfb_input_upload')
        if supplement and st.button('Merge CFB inputs'):
            try:
                uploaded=json.load(supplement);updated,matched,unmatched=apply_cfb_inputs(data,uploaded.get('teams',{}),ignore_nulls=True)
                if not matched: raise ValueError('No CFB input teams matched the loaded teams.')
                use_dataset(updated);rerun()
            except (ValueError,TypeError) as exc: st.error(str(exc))
        st.caption('Blank/null fields in this template preserve existing inputs. Use the team editor to clear a field. Manual FEI ratings keep their through-week until you refresh FEI explicitly. Export the full snapshot to keep edits between browser sessions.')
    st.subheader('Import or export a snapshot')
    upload=st.file_uploader('Upload model-input snapshot JSON (older builds)',type=['json'],key='snapshot_import')
    if upload and st.button('Load uploaded snapshot'):
        try:
            parsed=json.load(upload);use_dataset(parsed.get('snapshot',parsed));rerun()
        except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
    st.download_button('Export model inputs and settings',json.dumps(data,indent=2,allow_nan=False),file_name='football_model_snapshot.json',mime='application/json')
    st.caption('Keep the full JSON to restore inputs, weights, historical meetings and manual overrides in a later Streamlit session.')
    with st.expander('Edit the complete dataset JSON'):
        text=st.text_area('Teams, games, markets, parameters and sources',json.dumps(data,indent=2),height=350,key='dataset_json_editor')
        if st.button('Apply complete JSON'):
            try:use_dataset(json.loads(text));rerun()
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
    if st.button('Restore original 11-game snapshot'):
        use_dataset(json.loads((ROOT/'snapshot.json').read_text()));rerun()

with tabs[4]:
    with st.expander('Control guide in plain English',expanded=True):
        st.write('Projected score estimates how many points each team scores. Spread estimates the winning margin; +10 is a ten-point head start. Total is both scores added together. An ATS cover can happen even when the team loses the game.')
        labels={'by':'Yards per play weight','bs':'Success rate weight','shrink':'Efficiency shrinkage','q':'Baseline points per team','h':'Home-field margin','n':'Reference plays per team','y':'Reference yards per play','s':'Reference success %','x':'Reference explosiveness','bx':'Explosiveness weight','minimum_ev':'Minimum conditional EV %'}
        st.dataframe(pd.DataFrame([{'Control':labels.get(key,key.replace('_',' ').title()),'What it does and what moving it changes':value} for key,value in CONTROL_HELP.items()]),hide_index=True,width='stretch')
    st.subheader('CFB Colley, Massey and gradient boosting')
    st.code('Colley: Cii = 2 + games_i; Cij = −meetings_ij; bi = 1 + (wins_i−losses_i)/2; solve Cr=b\nMassey: minimize ||Xr − neutralized prior margins||², with mean rating zero per connected schedule\nBoosting features: prior-week efficiency, pace, turnovers, baseline margin/total, Massey gap, Colley gap and sample size\nFit two tree ensembles to actual margin/total minus the statistical core forecast\nSelect tree settings on validation seasons; cap margin correction at ±10 and total at ±15\nApply only targets improving validation RMSE; optional FEI/roster/injury/context terms remain separate')
    st.write('Massey is the default CFB point-rating method. Colley is a win/loss rating used as a learned feature, not a probability or a point spread. Disconnected Massey comparisons are withheld. League controls, top-two rankings and injury imports are separate. CFB look-ahead controls never modify NFL settings.')
    st.subheader('NFL training and reference fit')
    st.write('NFL Historical Training loads free nflverse seasons, summarizes eligible plays, and reconstructs only completed prior weeks. Two separate tree ensembles learn corrections to winning margin and combined score. EPA, dropbacks, rushing, CPOE, sacks, rest and regularized opponent ratings are NFL inputs. The validation period selects the correction; later test results are reported without choosing the model. NFL and college trained models have independent controls and formats.')
    st.subheader('Score equations')
    st.code('M = (offensive metric + opponent allowed metric) / 2\nF = (away plays + home plays) / (2 × reference plays)\nB = baseline + shrink × [bY×(Y−Yref) + bS×(S−Sref) + bX×(X−Xref)]\nT = turnover point value × retention × (away net TO − home net TO) / 4\nAway score = F×Baway + T − (home field+rest)/2 + injury away + weather/2\nHome score = F×Bhome − T + (home field+rest)/2 + injury home + weather/2\nFair away spread = home score − away score\nFair total = home score + away score\nConditional EV = net profit multiple × P(win) − P(loss)')
    st.write('Home field is zero at neutral venues. Extra travel is currently absorbed into home field. Turnover retention is a relative-rating proxy; it is not a separately estimated fumble recovery model.')
    st.subheader('CFB FEI, continuity and portal equations')
    st.code('C_i = original pace-adjusted efficiency forecast\nD = pace × reference possessions\nE_i = pace × baseline + D × (OFEI_i − DFEI_opponent)\nBlended efficiency = (1−wFEI) × C_i + wFEI × E_i\nd_i = 2^(−games_played_i / roster_half_life)\nReturning adjustment_i = bStarter × [d_i × (off_starters_i−reference) − d_opponent × (def_starters_opponent−reference)]\nNet portal rating_i = sum(incoming provider ratings) − sum(outgoing provider ratings)\nAway portal adjustment = bPortal × (d_away × net_away − d_home × net_home) / (2 × portal_scale)\nHome portal adjustment = −Away portal adjustment\nFinal score = blended efficiency + returning adjustment + portal adjustment + existing game modifiers')
    st.write('FEI is blended into the efficiency forecast; overall FEI and its ranks are not added again. The FEI baseline and possession conversion are modeling assumptions because the source rates non-garbage possessions. Positive DFEI lowers the opponent score. Portal is a coarse team-strength adjustment to margin and does not change the total.')
    st.write('Head-to-head uses up to five prior seasons from the schedule feeds. Past margins are converted to the current team perspective and approximately neutralized for home field. Age weights have a 2-year half-life; effective sample shrinkage is n/(n+8). Only 10% of a discrepancy capped at 10 points is applied by default. Old matchup results are a weak contextual prior because players, coaches, and team strength change.')
    st.subheader('Opponent-adjusted results and ATS equations')
    st.code('For each prior game: y = away points − home points + home field (0 at neutral sites)\nX has +1 for away, −1 for home\nResult ratings r = solve(XᵀX + λI, Xᵀy)\nEfficiency margin = away pace/FEI forecast − home pace/FEI forecast\nResult adjustment = wResult × [(rAway − rHome) − efficiency margin]\nSmoothed ATS rate = (wins + prior_games/2)/(wins + losses + prior_games)\nATS adjustment = clip(bATS × [away road rate − home home rate], −cap, cap)')
    st.write('Result ratings use only current-season completed games through the feature cutoff. They add an independent point-scale strength signal and can move forecasts toward either side. Default result blends are 50% CFB and 25% NFL, with at least 2 prior games per team. CFB defaults to standard Massey; ridge penalty 2 applies only to the optional regularized method and NFL. They are provisional. The ATS window is ten calendar years through the slate start; neutral games and unverified lines are excluded. Its default influence is 2 points per full rate difference, with a maximum ±0.5 margin-point effect. Both layers leave the total unchanged. Historical coach/roster changes and selective line coverage limit what ATS trends predict.')
    st.subheader('Schedule context equations')
    st.code('L_i = next major game within the time window, with current opponent below the major threshold\nW_i = previous completed win over a major opponent within the time window\nR_i = previous completed road loss, followed by a non-neutral home game within the time window\nΔMargin = clip(bL×(L_away−L_home) + bW×(W_away−W_home) + bR×(R_away−R_home), −cap, cap)\nΔTotal = clip(tL×(L_away+L_home) + tW×(W_away+W_home), −cap, cap)\nAway score contribution = (ΔMargin+ΔTotal)/2\nHome score contribution = (ΔTotal−ΔMargin)/2')
    st.write('Major opponents use a frozen neutral-strength proxy: shrink × [bY×(y−dy)+bS×(s−ds)+bX×(x−dx)]/2. CFB blends that proxy with reference possessions × (OFEI+DFEI) when dated FEI is eligible. Default major thresholds are 4 points above average for NFL and 8 for CFB. Explicit schedule major_game tags or per-game manual flags support rivalry context. Unknown inputs produce no automatic adjustment; future results are never used.')
    st.write('The default time window is 10 days and the combined margin/total adjustments are each capped at 2 points. Look-ahead and major-win weights start at zero; set signed effects to test a letdown or confidence scenario. Returning home after a verified road loss gets a provisional +0.5 margin-point boost, separate from normal home field, with no automatic change to the total. These effects need historical validation and are not multiplied by the score-calibration slope.')
    st.write('Default CFB priors: 25% FEI blend, 12 reference possessions, 0.20 points per extra returning starter, 1 portal point per 5 native net-rating units, and a 4-game roster half-life. Missing FEI disables the game’s FEI blend; missing roster inputs skip their terms. These values require chronological validation and may overlap with FEI preseason priors or current-season efficiency.')
    st.write('Normal residuals use integer push bins and half-point lines. The distribution does not model football scoring clusters at 3 and 7. Uncertainty and probabilities need out-of-sample calibration.')
    st.subheader('The path to prediction quality')
    st.markdown('''1. Store immutable, timestamped features and odds before kickoff, plus eventual results.
2. Fit separate NFL and CFB models with opponent adjustment, regularization and sample-size shrinkage.
3. Validate on later weeks/seasons. Compare margin and total MAE/RMSE with a timestamp-matched market baseline.
4. Calibrate cover/over probabilities and the score distribution on held-out games. Evaluate Brier score, log loss and calibration plots.
5. Select weights and thresholds on training/validation data; report final test results once. Rank future games only after measuring uncertainty and stability.
''')
    st.write('This package supplies independent CFB and NFL training, reviewed injuries, ATS/total trend screens and CLV logging. The included NFL reference fit uses 2016–2025 data; its total correction passed validation and its spread correction stayed disabled. Historical closing-market forecasts had smaller 2025 errors than this NFL fit. CFB still needs your historical data before fitting. The college preference profile is editable and separate from learned coefficients.')
    st.json(data.get('sources',{}))

with tabs[3]:
    cfb_clv_tab,nfl_clv_tab=st.tabs(['CFB CLV','NFL CLV'])
    with cfb_clv_tab:clv_panel(data,results,'CFB',include_uploads=False)
    with nfl_clv_tab:clv_panel(data,results,'NFL',include_uploads=False)
    st.subheader('Favorite and underdog diagnostics')
    st.write('These summaries use the best quoted spread per game. They identify concentration and compressed forecasts; they do not establish which side will outperform.')
    for lg in ('CFB','NFL'):
        min_ev=st.session_state.get('min_ev_'+lg,0.)/100
        rows=[];model_margins=[];market_margins=[]
        for r in results:
            if r['league']!=lg or 'away' not in r:continue
            spreads=[q for q in r['markets'] if q['kind']=='spread']
            if not spreads:continue
            q=max(spreads,key=lambda q:q['conditional_ev'])
            reference=r['game'].get('line_away')
            if reference is not None:model_margins.append(r['margin_away']);market_margins.append(-reference)
            rows.append({'Matchup':r['game']['away']+' @ '+r['game']['home'],'Quoted selection':q['label'],'Side':'Favorite' if q['line']<0 else 'Underdog' if q['line']>0 else 'Pick’em','Conditional EV %':100*q['conditional_ev'],'Qualifies':q['conditional_ev']>min_ev})
        st.markdown('**'+lg+'**')
        if rows:
            qualifying=[row for row in rows if row['Qualifies']]
            st.write(f"Qualifying spread signals: {sum(row['Side']=='Favorite' for row in qualifying)} favorites, {sum(row['Side']=='Underdog' for row in qualifying)} underdogs, {sum(row['Side']=='Pick’em' for row in qualifying)} pick’em.")
            if len(model_margins)>1 and pstdev(market_margins)>0:
                ratio=pstdev(model_margins)/pstdev(market_margins)
                st.write(f"Margin standard deviation: model {pstdev(model_margins):.2f}, market {pstdev(market_margins):.2f}; ratio {ratio:.2f}.")
                if ratio<.85:st.warning('Model margins are compressed relative to this slate’s market. Check opponent strength, input quality and historical calibration before interpreting the underdog concentration.')
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        else:st.caption('No usable spread quotes loaded.')
    st.subheader('Learn calibration from archived pregame forecasts')
    st.write('Archive forecasts before kickoff. After the games finish, load their results and import your archive here. The fitter requires at least 80 distinct completed games per league with the same model settings, uses older games for training and later games for validation, and applies only corrections that improve validation RMSE.')
    archived=archive(data,results)
    st.download_button('Archive current pregame forecasts',json.dumps(archived,indent=2),file_name='pregame_forecasts.json',mime='application/json',disabled=not archived['forecasts'])
    if not archived['forecasts']:st.caption('No future games with verified kickoff timestamps are loaded. Refresh schedules before archiving.')
    calibration_league=st.selectbox('Calibration league',['CFB','NFL'])
    if st.session_state.get('calibration_archive'):
        try:
            payload=st.session_state['calibration_archive']; records=attach_results(payload['forecasts'],data)
            st.download_button('Download archive with available results',json.dumps({'format':'pregame-calibration-v1','forecasts':records},indent=2),file_name='forecasts_with_results.json',mime='application/json')
            if st.button('Fit and validate calibration'):
                report=candidate(records,calibration_league,signature(data))
                st.session_state['calibration_report']=report
                st.session_state['calibration_signature']=signature(data)
        except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
    report=st.session_state.get('calibration_report')
    if report and report['league']==calibration_league:
        st.json(report)
        improvements=[target for target in ('margin','total') if report['metrics'][target]['improves']]
        if improvements and st.session_state.get('calibration_signature')==signature(data):
            if st.button('Apply validated corrections'):
                for target in improvements:
                    for key,value in report['coefficients'][target].items():data['parameters'][calibration_league][target+'_'+key]=value
                reset_widgets();rerun()
        else:st.caption('No correction qualifies, or the model settings changed. Refit after reviewing the data.')

# Render setup, calibration and method before optional pick/trend/prop panels.
# Streamlit evaluates collapsed expanders and unselected tabs as well.
with tabs[0]:
    render_slate('CFB')
with tabs[1]:
    render_slate('NFL')
    attd_panel(data)

# Persist controls, histories, reports, overrides and tracking after all panels.
saved=autosave()
if st.session_state.get('_storage_error'):
    setup_status.error(st.session_state['_storage_error']);save_badge.warning('Workspace save needs attention — open Setup & Saved Data.')
elif st.session_state.get('_store'):
    setup_status.success(st.session_state.get('_storage_message','Connected'))
    save_badge.caption('Saved workspace connected · '+st.session_state.get('_storage_message','Loaded'))
else:
    save_badge.caption('Setup & Saved Data lists all inputs. Connect cloud storage once to keep them between visits.')
