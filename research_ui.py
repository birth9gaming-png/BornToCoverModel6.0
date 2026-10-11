"""Historical training, reviewed injury inputs and quote tracking panels."""
from __future__ import annotations
import copy
import json
from datetime import datetime,timezone
import zipfile
import pandas as pd
import streamlit as st
from workspace_runtime import rerun
from historical_data import import_files,coverage,fetch_seasons
from historical_training import build_rows,fit_boost,dataset_hash
from cfb_ml import signature,correction,FEATURES
from injuries import read_report,review_rows,apply_to_games,STATUSES
from clv import capture,read_closes,summaries


def historical_panel(data,secret,reset_widgets):
    st.subheader('CFB Historical Training')
    st.write('Upload years of game-level data here, or fetch selected seasons using your CollegeFootballData key. Loading a key or refreshing this week’s slate does not train the model.')
    st.dataframe(pd.DataFrame([
        {'File':'Games / results','CFBD choice':'Games → Returns historical game data','Purpose':'Final scores, kickoff, week and venue; required'},
        {'File':'Team GAME stats','CFBD choice':'Games → Returns team box score statistics by game','Purpose':'Yards, play attempts and turnovers; required'},
        {'File':'Advanced GAME stats','CFBD choice':'Stats → Returns advanced team statistics aggregated by game','Purpose':'Success rate and explosiveness; required'},
        {'File':'Betting lines','CFBD choice':'Betting → Returns betting lines','Purpose':'Optional market benchmark; not assumed to be closing quotes'},
    ]),hide_index=True,width='stretch')
    uploads=st.file_uploader('Upload historical CFB files (select multiple files, or one ZIP)',type=['csv','json','zip'],accept_multiple_files=True,key='history_upload')
    merge=st.checkbox('Merge with history already loaded in this session',value=True,key='history_merge')
    if st.button('Import and check historical files',disabled=not uploads):
        try:
            existing=st.session_state.get('cfb_historical_dataset') if merge else None
            dataset=import_files([(u.name,u.getvalue()) for u in uploads],existing)
            st.session_state['cfb_historical_dataset']=dataset
            st.session_state.pop('cfb_training_result',None);st.success('Historical files imported. Review coverage below before training.')
        except (ValueError,TypeError,zipfile.BadZipFile,RuntimeError) as exc:st.error(str(exc))
    with st.expander('Fetch history with my CFBD API key'):
        st.caption('The free CFBD plan currently includes historical games, team stats, advanced metrics and lines, subject to endpoint coverage and monthly limits. [Check current limits](https://collegefootballdata.com/api-tiers). The key is used for requests and is never exported.')
        a,b=st.columns(2)
        years=a.multiselect('Historical seasons',list(range(2004,datetime.now().year)),default=list(range(2016,2026)),key='history_years')
        include_lines=b.checkbox('Also fetch historical betting lines',value=True,key='history_lines')
        history_key=st.text_input('CFBD key for historical download',value=st.session_state.get('cfb_api_live',secret('CFBD_API_KEY')),type='password',key='history_key')
        st.caption(f'Budget about {len(years)*(24 if include_lines else 23)} requests for this selection. Actual count is 3 season requests including AP polls + one per regular-season week, plus optional lines. Cached reads reduce network calls. Start with one season to check coverage, then fetch the remaining years.')
        if st.button('Fetch selected historical seasons',disabled=not years or not history_key):
            note=st.empty()
            with st.spinner('Downloading historical seasons…'):
                try:
                    files,info=fetch_seasons(years,history_key,progress=note.caption,include_lines=include_lines)
                    if files:
                        st.session_state['cfb_historical_dataset']=import_files(files,st.session_state.get('cfb_historical_dataset') if merge else None)
                        st.session_state.pop('cfb_training_result',None)
                    st.caption(f"Completed {info['requests_including_cached_reads']} request/cache reads.")
                    for error in info['errors']:st.warning(error)
                except (ValueError,TypeError,RuntimeError) as exc:st.error(str(exc))
    dataset=st.session_state.get('cfb_historical_dataset')
    if not dataset:
        if data.get('cfb_boost_model'):
            st.info('A saved CFB fit is present. Raw history is separate and is needed for the trend finder or retraining. Check the College Football projection breakdown to confirm whether the saved fit is active.')
        else:
            st.info('No historical training files loaded yet. Restore cfb_historical_dataset.json here, or fetch seasons once and export them. To reuse a saved fit, load its full snapshot in Setup & Saved Data or its trained-model JSON below.')
    else:
        st.subheader('Imported history coverage')
        st.dataframe(pd.DataFrame(coverage(dataset)),hide_index=True,width='stretch')
        if dataset['issues']:st.warning('Some rows are incomplete, ambiguous or conflicting. Those inputs are withheld.');st.dataframe(pd.DataFrame(dataset['issues']),hide_index=True,width='stretch')
        with st.expander('File import details'):st.dataframe(pd.DataFrame(dataset['files']),hide_index=True,width='stretch')
        st.download_button('Export checked historical dataset',json.dumps(dataset,allow_nan=False),file_name='cfb_historical_dataset.json',mime='application/json')
        st.caption('Save this export and upload it here to restore your history after a Streamlit restart. Cloud saving and complete backups include this checked history; a model-input snapshot alone does not.')
        a,b=st.columns(2)
        validation_start=a.number_input('First validation season',min_value=2005,max_value=2100,value=None if 'training_validation' in st.session_state else 2023,step=1,key='training_validation')
        test_start=b.number_input('First untouched test season',min_value=2006,max_value=2100,value=None if 'training_test' in st.session_state else 2025,step=1,key='training_test')
        st.write(f'Train: before {validation_start}. Choose tree settings and uncertainty: {validation_start}–{test_start-1}. Final test: {test_start} onward. Each game uses completed earlier weeks in the same season; the first two weeks and incomplete histories are withheld.')
        st.caption('This trains the play-stat/result-rating core. Historical FEI snapshots, roster, injury, ATS and discretionary schedule effects are not supplied by these four files, so those live additions remain separate assumptions. Historical exports can contain later provider revisions; this backtest is retrospective, not proof of what was published at each kickoff.')
        if st.button('Build prior-week features and train CFB gradient boosting'):
            bar=st.progress(0.,text='Building historical features')
            try:
                fingerprint=dataset_hash(dataset);core=signature(data['parameters'])
                cache_key=(fingerprint,core)
                cached=st.session_state.get('cfb_training_rows')
                if cached and cached['key']==cache_key:rows,skipped=cached['rows'],cached['skipped']
                else:
                    rows,skipped=build_rows(dataset,data['parameters'],progress=lambda value,text:bar.progress(value,text=text))
                    st.session_state['cfb_training_rows']=dict(key=cache_key,rows=rows,skipped=skipped)
                with st.spinner('Fitting trees and evaluating later seasons…'):
                    model,report,forecasts=fit_boost(rows,data['parameters'],int(validation_start),int(test_start),fingerprint)
                report['withheld_games']=skipped
                st.session_state['cfb_training_result']=dict(model=model,report=report,forecasts=forecasts)
                st.success('Fit complete. Review validation and final test metrics before applying.')
            except (ValueError,TypeError,KeyError,ImportError) as exc:st.error(str(exc))
            finally:bar.empty()
    fitted=st.session_state.get('cfb_training_result')
    if fitted:
        report=fitted['report'];st.subheader('Chronological validation and calibration')
        st.write('Lower MAE/RMSE means smaller score errors. Bias is actual minus prediction; positive bias means the model predicted too few points. A target is enabled only if its gradient-boosting correction beats the baseline on validation RMSE. Test results do not select a candidate.')
        rows=[]
        for target,details in report['targets'].items():
            for split,values in details['metrics'].items():
                rows.append(dict(target=target,split=split,games=report['counts'][split],enabled=details['enabled'],baseline_MAE=values['baseline']['mae'],selected_MAE=values['selected']['mae'],baseline_RMSE=values['baseline']['rmse'],selected_RMSE=values['selected']['rmse'],selected_bias=values['selected']['bias']))
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        for target,counts in report.get('probability_test_coverage',{}).items():
            withheld=counts['non_half_point_lines']+counts['invalid_lines']
            if withheld:
                st.caption(f"{target.title()} probability checks scored {counts['scored_games']} test games; excluded {withheld} invalid or non-whole/half-point reference lines. {counts['missing_lines']} games had no line. Multi-book medians can produce quarter points; eligible games still enter training and finite medians remain in the point-error comparison.")
        for target,values in report.get('probability_test',{}).items():
            st.markdown('**'+target.title()+' probability calibration on test games**')
            st.caption(f"Three-outcome Brier score: {values['three_outcome_brier']:.4f}; log loss: {values['log_loss']:.4f}; games: {values['games']}. Lower is better. These score win/push/loss predictions against historical reference lines; the lines are not verified decision-time or closing prices.")
            st.dataframe(pd.DataFrame(values['reliability']),hide_index=True,width='stretch')
        with st.expander('Full fit report, market comparison and withheld games'):st.json(report)
        qualifies=any(v['enabled'] for v in report['targets'].values())
        unchanged=signature(data['parameters'])==report['core_signature']
        if st.button('Apply validated CFB model and validation uncertainty',disabled=not qualifies or not unchanged):
            data['cfb_boost_model']=copy.deepcopy(fitted['model']);data['cfb_training_report']=copy.deepcopy(report)
            for target,key in [('margin','sigma_spread'),('total','sigma_total')]:data['parameters']['CFB'][key]=report['targets'][target]['validation_sigma']
            reset_widgets();rerun()
        if not unchanged:st.info('Core controls changed since this fit. Restore those controls or train again before applying it.')
        if not qualifies:st.caption('No target improved validation RMSE; the original core remains selected. Review data quality or collect more seasons.')
        st.download_button('Download trained CFB model',json.dumps({'model':fitted['model'],'report':report},allow_nan=False),file_name='cfb_trained_model.json',mime='application/json')
        forecasts=[dict(r,features=json.dumps(r['features']),split=split) for split,group in fitted['forecasts'].items() for r in group]
        st.download_button('Download historical backtest rows',pd.DataFrame(forecasts).to_csv(index=False),file_name='cfb_backtest_rows.csv',mime='text/csv')
    st.subheader('Restore a trained CFB model')
    uploaded=st.file_uploader('Import trained model JSON',type=['json'],key='trained_model_upload')
    if st.button('Load trained CFB model',disabled=not uploaded):
        try:
            if len(uploaded.getvalue())>10*1024*1024:raise ValueError('Model exceeds 10 MB.')
            payload=json.loads(uploaded.getvalue());model=payload['model']
            correction(model,dict.fromkeys(FEATURES,0.))
            if model.get('core_signature')!=signature(data['parameters']):raise ValueError('Model was fitted with different CFB core settings. Restore its original full snapshot or match the original settings first.')
            if isinstance(payload.get('report'),dict):
                uncertainty={}
                for target,key in [('margin','sigma_spread'),('total','sigma_total')]:
                    value=float(payload['report']['targets'][target]['validation_sigma'])
                    if not 0<value<100:raise ValueError('Invalid validation uncertainty.')
                    uncertainty[key]=value
                data['cfb_training_report']=payload['report'];data['parameters']['CFB'].update(uncertainty)
            data['cfb_boost_model']=model
            reset_widgets();rerun()
        except (ValueError,TypeError,KeyError,IndexError) as exc:st.error(str(exc))


def injury_panel(data,league):
    with st.expander(league+' injury reports and reviewed point impacts'):
        st.write('Upload a CSV, JSON, text report or readable PDF. The app reads player statuses; you review team matching, publication times and the score impact. Player importance is not inferred from a name or injury status alone.')
        if league=='CFB':st.markdown('[Official SEC availability reports](https://www.secsports.com/fbreports). Coverage varies by conference and school. This version does not claim an automatic complete FBS injury feed.')
        teams=[t.get('name',tid) for tid,t in data['teams'].items() if t.get('league')==league]
        fallback=st.selectbox('Team for a single-team PDF/text report',teams,index=None,key='injury_team_'+league)
        upload=st.file_uploader('Upload '+league+' injury report',type=['csv','json','pdf','txt'],key='injury_upload_'+league)
        st.download_button('Download '+league+' injury CSV template','team,player,status,report_as_of,point_impact\n',file_name=league.lower()+'_injuries.csv',mime='text/csv',key='injury_template_'+league)
        st.caption('report_as_of needs the report’s actual publication time with timezone, e.g. 2026-10-09T18:00:00-04:00. point_impact is a reviewed reduction in that team’s expected score; start at 0 until you can justify a value.')
        if st.button('Read '+league+' injury file',disabled=not upload,key='injury_read_'+league):
            try:
                rows,ignored=read_report(upload.name,upload.getvalue(),teams,fallback)
                st.session_state['injury_draft_'+league]=rows
                if ignored:st.caption(f'{ignored} unreadable or ambiguous rows were withheld.')
            except Exception as exc:st.error('Could not read the injury report: '+str(exc))
        draft=st.session_state.get('injury_draft_'+league)
        if draft:
            edited=st.data_editor(pd.DataFrame(draft),hide_index=True,num_rows='dynamic',width='stretch',key='injury_review_'+league,
                                  column_config={'status':st.column_config.SelectboxColumn(options=list(STATUSES)),'point_impact':st.column_config.NumberColumn(min_value=-20.,max_value=0.,step=.25)})
            checked=st.checkbox('I checked the team, player status, report publication time and point impact',key='injury_checked_'+league)
            if st.button('Save reviewed '+league+' injury rows',disabled=not checked,key='injury_save_'+league):
                approved,errors=review_rows(edited.to_dict('records'),data,league)
                if errors:
                    for error in errors:st.error(error)
                else:
                    existing=[r for r in data.get('injury_reports',[]) if r.get('league')!=league]
                    data['injury_reports']=existing+approved;st.success(f'Saved {len(approved)} reviewed player rows. Apply below to set game adjustments.')
        reports=[r for r in data.get('injury_reports',[]) if r.get('league')==league]
        if reports:
            st.dataframe(pd.DataFrame(reports),hide_index=True,width='stretch')
            if st.button('Apply reviewed injuries to loaded '+league+' games',key='injury_apply_'+league):
                try:
                    apply_injuries(data,league);rerun()
                except ValueError as exc:st.error(str(exc))
        st.caption('Only reviewed reports published before kickoff and within 96 hours of it are eligible. Latest status per player wins; Available cancels that player’s adjustment. Effects are capped at −10 score points per team. Missing or stale reports mean unknown coverage. Applying reports replaces the injury score adjustments for games covered by those rows.')


def apply_injuries(data,league):
    apply_to_games(data,league)


def clv_panel(data,results,league='CFB',include_uploads=True):
    stem=league.lower();ledger_key=stem+'_clv_ledger';closing_key=stem+'_closing_quotes';fmt=stem+'-clv-v1'
    st.subheader(league+' Closing Line Value (CLV)')
    st.write('Record today’s two highest conditional-EV '+league+' signals before kickoff, then import the same book’s final pregame quote. Positive point CLV means your recorded line was more favorable. CLV measures price movement; it does not by itself establish profitability.')
    eligible=capture(data,results,league=league)
    if st.button('Log current top two '+league+' signals',disabled=not eligible,key=stem+'_clv_log'):
        ledger=data.setdefault(ledger_key,[])
        for row in eligible:
            identity=tuple(row[k] for k in ('game_id','book','market','side','line','quote_as_of'))
            if not any(tuple(old[k] for k in ('game_id','book','market','side','line','quote_as_of'))==identity for old in ledger):ledger.append(row)
        st.success('Recorded eligible pregame quotes and model settings.')
    if not eligible:st.caption('Refresh the slate with current named-book odds to enable logging. Quotes need verified kickoff and update timestamps, must be less than six hours old, and must precede kickoff. Manual scenario odds are excluded.')
    ledger=data.get(ledger_key,[])
    st.download_button('Export '+league+' CLV ledger',json.dumps({'format':fmt,'ledger':ledger,'closing_quotes':data.get(closing_key,[])},indent=2),file_name=stem+'_clv_ledger.json',mime='application/json',key=stem+'_clv_export')
    if include_uploads:clv_uploads(data,league)
    if ledger:
        table=summaries(ledger,data.get(closing_key,[]));st.dataframe(pd.DataFrame(table),hide_index=True,width='stretch')
        valid=[r for r in table if r.get('points') is not None]
        if valid:
            a,b,c=st.columns(3);a.metric('Compared quotes',len(valid));b.metric('Mean point CLV',f"{sum(r['points'] for r in valid)/len(valid):+.2f}");c.metric('Positive CLV',f"{sum(r['points']>0 for r in valid)/len(valid):.0%}")
    st.code('Spread CLV = recorded handicap − closing handicap\nOver CLV = closing total − recorded total\nUnder CLV = recorded total − closing total')
    st.caption('This uses observed closing snapshots, not a guarantee of the exact last quote. Price CLV is shown only when the line is unchanged and includes bookmaker margin. Closing information stays out of historical prediction features. Export the ledger or full snapshot to retain records after a restart.')


def clv_uploads(data,league):
    stem=league.lower();ledger_key=stem+'_clv_ledger';closing_key=stem+'_closing_quotes';fmt=stem+'-clv-v1'
    restore=st.file_uploader('Restore '+league+' CLV ledger JSON',type=['json'],key=stem+'_clv_restore')
    if st.button('Restore saved '+league+' CLV ledger',disabled=not restore,key=stem+'_clv_restore_apply'):
        try:
            payload=json.loads(restore.getvalue())
            if payload.get('format')!=fmt or not isinstance(payload.get('ledger'),list):raise ValueError('Use the exported '+fmt+' ledger.')
            if any(row.get('league')!=league for row in payload['ledger']):raise ValueError('Ledger belongs to a different league.')
            summaries(payload['ledger'],payload.get('closing_quotes',[]))
            data[ledger_key]=payload['ledger'];data[closing_key]=payload.get('closing_quotes',[]);rerun()
        except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
    st.download_button('Download '+league+' closing-quote CSV template','game_id,book,market,side,line,odds,quoted_at\n',file_name=stem+'_closing_quotes.csv',mime='text/csv',key=stem+'_clv_template')
    st.caption('Copy game_id and book from the ledger. market: spread/total; side: away/home/over/under. Use the selected team’s own signed handicap. quoted_at requires a timezone. Closing snapshots must be after capture and within 60 minutes before kickoff; arbitrary historical line exports are not treated as closing quotes.')
    upload=st.file_uploader('Import '+league+' closing quotes CSV or JSON',type=['csv','json'],key=stem+'_clv_closes')
    if st.button('Add '+league+' closing snapshots',disabled=not upload,key=stem+'_clv_add'):
        try:
            data.setdefault(closing_key,[]).extend(read_closes(upload.name,upload.getvalue()));st.success('Closing quotes loaded; unmatched or invalid timing will be shown below.')
        except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
