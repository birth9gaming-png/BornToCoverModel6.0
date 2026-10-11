"""One place for readiness, one-time setup, and every file upload."""
from __future__ import annotations
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
from model import METRICS
from matchup_history import timestamp
from persistence import make_bundle,encode,validate_bundle,StorageError
from app_lifecycle import preferences
from auto_injuries import settings as injury_settings,refresh as refresh_injuries,eligible_reports
from workspace_runtime import autosave,load_saved,rerun

SECRET_TEMPLATE='''CFBD_API_KEY = "your-college-football-data-key"
ODDS_API_KEY = "your-odds-api-key"
R2_ACCOUNT_ID = "your-32-character-cloudflare-account-id"
R2_ACCESS_KEY_ID = "your-r2-access-key-id"
R2_SECRET_ACCESS_KEY = "your-r2-secret-access-key"
R2_BUCKET = "football-model"'''


def readiness(data,session,secret,results,now=None):
    now=now or datetime.now(timezone.utc);rows=[]
    def add(item,need,status,action):rows.append({'Input or setting':item,'Needed for':need,'Status':status,'Action':action})
    store=session.get('_store')
    cloud='Error: '+session['_storage_error'] if session.get('_storage_error') else ('Connected' if store else 'Not connected')
    add('Persistent saved workspace','Remember everything',cloud,'One-time cloud setup below; no weekly upload')
    add('Personal app access','Private login (once)','Set in host Sharing settings','Only specific people can view this app; use your own account')
    for key,purpose in [('CFBD_API_KEY','Automatic CFB refresh'),('ODDS_API_KEY','ATS / totals / ATTD prices')]:
        add(key,purpose,'Saved in host' if secret(key) else 'Missing from host Secrets','Save once in Streamlit Settings → Secrets')
    add('Automatic refresh','This week’s live data','On' if preferences(data)['auto_refresh'] else 'Off','On by default; checks on visits and while app is open')
    for lg in ('CFB','NFL'):
        games=[g for g in data['games'] if g.get('league')==lg];future=[]
        for g in games:
            try:
                if timestamp(g.get('kickoff'))>now:future.append(g)
            except ValueError:pass
        try:fresh=now-timestamp(data.get('as_of'))<timedelta(hours=24)
        except ValueError:fresh=False
        state='Frozen example' if data.get('mode')=='snapshot' else ('Current' if fresh and future and lg not in data.get('stale_leagues',{}) else 'Needs refresh / no upcoming games')
        add(lg+' schedules and statistics','Score projections',state+f' · {len(future)} upcoming','Automatic; manual refresh below if needed')
        involved={g[side] for g in games for side in ('away','home')}
        missing=sum(any(data['teams'].get(tid,{}).get(k) is None for k in METRICS) for tid in involved)
        add(lg+' core team statistics','Score projections',f'{len(involved)-missing}/{len(involved)} teams complete','Automatic provider load; inspect missing team fields')
        market_games=0
        for g in future:
            try:
                if any(q.get('book') and q.get('updated_at') and timedelta(0)<=now-timestamp(q['updated_at'])<=timedelta(hours=6) for q in g.get('quotes',[])):market_games+=1
            except ValueError:pass
        add(lg+' current named-book odds','Price-based picks',f'{market_games}/{len(future)} upcoming games','Automatic with ODDS_API_KEY; absent quotes mean no current value')
        history=session.get(lg.lower()+'_historical_dataset')
        years=sorted({r.get('season') for r in (history or {}).get('games',[]) if r.get('season') is not None})
        add(lg+' checked historical dataset','Retraining / historical trends (once)',('Loaded · '+(', '.join(map(str,years)) or 'checked dataset')) if history else 'Not loaded','Restore checked-history JSON or fetch once in history section')
        model=data.get(lg.lower()+'_boost_model');active=sum(r.get('gradient_boosting',{}).get('active',False) for r in results if r.get('league')==lg)
        fit_state=f'Active on {active} projections' if active else ('Loaded; inactive — inspect fit/settings/inputs' if model else 'No trained model loaded')
        add(lg+' trained model','Learned corrections (once)',fit_state,'Restore full backup / trained model; or train and apply once')
        config=injury_settings(data);check=data.get('auto_injury_checks',{}).get(lg,{})
        state='Off' if not(config['enabled'] and config[lg]) else check.get('state','Waiting for a live upcoming slate')
        add(lg+' automatic injury coverage','Availability / injury scenarios',state+' · '+str(check.get('matched_players',0))+' reports','Free automatic checks; partial coverage, never assumed healthy')
        add(lg+' reviewed injury files','Optional manual override',str(sum(r.get('league')==lg for r in data.get('injury_reports',[])))+' rows','Optional; upload only to override or supplement a listing')
        add(lg+' CLV ledger / closing quotes','Optional tracking',str(len(data.get(lg.lower()+'_clv_ledger',[])))+' logged quotes','Optional; restore a ledger and add closes below')
    add('CFB FEI, portal, returning starters','Optional CFB terms','See team coverage below','FEI/portal refresh automatically; starters require verified input')
    add('NFL player roles / active-game logs','Anytime touchdown scorers',f"{len(data.get('nfl_player_candidates',[]))} players / {len(data.get('nfl_player_game_logs',[]))} logs",'Automatic NFL refresh; optional provider JSON below')
    add('NFL ATTD prices','Price-based ATTD value',str(len(data.get('nfl_attd_quotes',[])))+' quotes','Automatic for qualifying games; provider access/quota apply')
    add('NFL wind forecasts','Weather adjustment',str(len(data.get('nfl_weather_inputs',[])))+' rows','Automatic for exposed venues; optional provider override')
    add('NFL official DVOA / red-zone DVOA / adjusted line yards','Optional NFL terms',str(len(data.get('nfl_team_inputs',[])))+' dated team input rows','Optional dated provider JSON; unavailable terms are skipped')
    add('Player injury values / absence baselines','Optional injury refinement',str(len(data.get('injury_player_values',[])))+' overrides','Optional JSON below; defaults are unfitted scenario priors')
    add('Calibration archive','Optional later-game calibration','Loaded' if session.get('calibration_archive') else 'Not loaded','Optional archive upload below; never required to open projections')
    return rows


def injury_overview(data,league):
    config=injury_settings(data);check=data.get('auto_injury_checks',{}).get(league,{})
    with st.expander(league+' automatic injury checks and freshness'):
        if not config['enabled'] or not config[league]:st.info('Automatic '+league+' injury checks are off.');return
        st.write(check.get('state','Waiting for a live upcoming slate')+' · '+str(check.get('source') or 'no current source'))
        st.caption('Last check: '+str(check.get('checked_at','Not checked yet'))+'. Public listings are partial; no listed player does not mean the team is healthy.')
        rows={}
        for game in data['games']:
            if game.get('league')==league:
                for row in eligible_reports(data,game):rows[(row['team_id'],row['player'])]=row
        if rows:
            st.dataframe(pd.DataFrame([{k:r.get(k) for k in ('team_id','player','position','status','report_updated_at','source')} for r in rows.values()]),hide_index=True,width='stretch')
        else:st.caption('No currently eligible reports for the upcoming loaded games.')
        for note in check.get('notes',[]):st.caption(note)
        st.caption('Free NFL reports use ESPN with Covers as a fallback. CFB uses Covers. Old, unmatched and post-kickoff reports are withheld. Reviewed uploads and all settings are in Setup & Saved Data.')


def panel(data,secret,reset_widgets,results):
    original_preferences=preferences(data);original_injuries=injury_settings(data)
    st.subheader('Setup & Saved Data')
    st.write('Everything needed to load, restore or improve this workspace is listed here. Optional inputs can remain missing. Your saved models and toggles are reused on later visits.')
    status=st.empty()
    st.dataframe(pd.DataFrame(readiness(data,st.session_state,secret,results)),hide_index=True,width='stretch')
    with st.expander('Cloud save, complete backup and one-time setup',expanded=True):
        st.write('Storage: '+st.session_state.get('_store_label','Not connected'))
        if st.session_state.get('_storage_error'):st.error(st.session_state['_storage_error'])
        elif st.session_state.get('_store'):st.success(st.session_state.get('_storage_message','Connected; automatic save runs after this page finishes.'))
        else:st.info('Connect private storage once to remember your data after the host restarts.')
        st.caption('Last successful save this session: '+st.session_state.get('_saved_at','No new changes saved this session'))
        guide=Path(__file__).with_name('personal_cloud_setup.md')
        if guide.exists():st.download_button('Download personal cloud setup instructions',guide.read_text(),file_name=guide.name,mime='text/markdown')
        st.markdown('1. Update your existing GitHub app with every file in Prototype 11.\n2. In Streamlit **Settings → Sharing**, choose **Only specific people can view this app** and keep access limited to your own account.\n3. Create a private **Standard** Cloudflare R2 bucket and a bucket-scoped **Object Read & Write** API token.\n4. Add the settings below to **Streamlit app Settings → Secrets**, then reboot the app.\n5. Restore your CFB history and applied model once, or import a complete backup. Wait for **Saved automatically**. Later visits load saved data and refresh upcoming games automatically.')
        st.code(SECRET_TEMPLATE,language='toml')
        st.caption('Keys belong in host Secrets and never in GitHub, uploads or backups. R2 has a free allowance, with charges beyond it. Streamlit can sleep when idle; reopen/wake it and it loads your saved workspace. Refreshes run on visits and while the app is open, not while the host sleeps.')
        st.download_button('Download complete workspace backup',encode(make_bundle(data,st.session_state)),file_name='football_complete_workspace.json',mime='application/json')
        uploaded=st.file_uploader('Restore complete workspace backup',type=['json'],key='workspace_restore')
        if st.button('Restore complete backup',disabled=not uploaded,key='workspace_restore_apply'):
            try:
                payload=validate_bundle(json.loads(uploaded.getvalue()))
                st.session_state['_pending_workspace']=dict(payload=payload,cloud=False);rerun()
            except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
        a,b=st.columns(2)
        if a.button('Save everything now',disabled=not st.session_state.get('_store'),key='workspace_save'):
            if autosave(force=True):st.success('All current data and settings saved.')
            else:st.error(st.session_state.get('_storage_error','Saving is paused; resolve the connection or reload a valid save.'))
        if b.button('Reload saved workspace',disabled=not st.session_state.get('_store'),key='workspace_reload'):
            try:load_saved(reset_widgets);rerun()
            except (StorageError,ValueError,TypeError) as exc:st.error(str(exc))
        st.caption('A reload replaces this browser’s changes with the saved copy. Download a complete backup first if you want to retain unsaved changes. Concurrent browser saves are checked before replacing the latest copy.')
    with st.expander('Automatic refresh and injury settings',expanded=True):
        prefs=preferences(data);data['app_preferences']=prefs
        prefs['auto_refresh']=st.checkbox('Automatically refresh upcoming schedules, statistics and odds',value=prefs['auto_refresh'],key='app_pref_auto')
        prefs['leagues']=st.multiselect('Automatically loaded leagues',['NFL','CFB'],default=prefs['leagues'],key='app_pref_leagues') or ['NFL','CFB']
        prefs['regions']=st.selectbox('Automatic odds region',['us','us2','uk','eu','au'],index=['us','us2','uk','eu','au'].index(prefs['regions']),key='app_pref_regions')
        prefs['refresh_minutes']=st.number_input('Minimum minutes between full refreshes',min_value=15,max_value=1440,value=int(prefs['refresh_minutes']),step=15,key='app_pref_interval')
        config=injury_settings(data);data['auto_injury_settings']=config
        for key,label in [('enabled','Automatically check injuries'),('NFL','Check NFL injuries'),('CFB','Check college football injuries'),('score_adjustments','Factor eligible injuries into projected scores')]:
            config[key]=st.checkbox(label,value=config[key],key='auto_injury_'+key)
        config['refresh_minutes']=st.number_input('Minimum minutes between injury checks',min_value=5,max_value=1440,value=int(config['refresh_minutes']),step=5,key='auto_injury_interval')
        config['questionable_absence']=st.slider('Questionable: assumed absence probability',0.,1.,float(config['questionable_absence']),step=.05,key='auto_injury_questionable')
        config['doubtful_absence']=st.slider('Doubtful: assumed absence probability',0.,1.,float(config['doubtful_absence']),step=.05,key='auto_injury_doubtful')
        st.caption('These probabilities and position values are editable assumptions. NFL score terms require a recent verified snap baseline or an explicit override. CFB uses a capped, conservative role proxy and skips continuing long-term absences. The projection breakdown shows each applied impact.')
        if st.button('Check free injury sources now',key='injury_live_refresh'):
            with st.spinner('Checking public availability listings…'):refresh_injuries(data,force=True)
            rerun()
    with st.expander('Optional player injury value overrides'):
        tid=next((tid for tid,t in data['teams'].items() if t.get('league')=='NFL'),'KC')
        example={'players':[{'league':'NFL','team_id':tid,'player':'Player Full Name','points':2.0,'baseline_absence':0.0}]}
        st.download_button('Download injury value override example',json.dumps(example,indent=2),file_name='injury_player_values.json',mime='application/json')
        st.caption('points is that player’s scoring-unit effect if absent (0–10); baseline_absence is the share of recent games already missed (0–1). A defensive absence adds to the opponent score. Values replace the generic role prior. Verify the team, player and baseline before import.')
        upload=st.file_uploader('Upload injury player values JSON',type=['json'],key='injury_values_upload')
        if st.button('Apply injury player values',disabled=not upload,key='injury_values_apply'):
            try:
                players=json.loads(upload.getvalue())['players']
                if not isinstance(players,list):raise ValueError('players must be a list.')
                for p in players:
                    if p['team_id'] not in data['teams'] or data['teams'][p['team_id']]['league']!=p['league'] or not p['player'].strip():raise ValueError('Each player needs a loaded, matching team and league.')
                    if not 0<=float(p['points'])<=10 or not 0<=float(p['baseline_absence'])<=1:raise ValueError('Use points 0–10 and baseline_absence 0–1.')
                encode(players);data['injury_player_values']=players;rerun()
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
    if preferences(data)!=original_preferences or injury_settings(data)!=original_injuries:rerun()
    return status
