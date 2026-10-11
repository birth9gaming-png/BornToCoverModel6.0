"""NFL history, training, calibration, and portable model controls."""
from __future__ import annotations
import copy
import json
import zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from workspace_runtime import rerun
from nfl_history import import_files,coverage,fetch_years,build_rows,dataset_hash
from nfl_ml import signature,correction
from historical_training import fit_boost


def historical_panel(data,reset_widgets):
    st.subheader('NFL Historical Training')
    st.write('Download past NFL seasons for free, or upload nflverse files. The downloader converts play-by-play into compact game summaries before building statistics available before each forecast week.')
    with st.expander('Included NFL history and reference fit'):
        st.caption('The package includes checked 2016–2025 regular-season history and a reference fit: 2016–2022 training, 2023–2024 validation, 2025 testing. The total correction qualified; the spread correction stayed disabled. The 2025 model errors were larger than the closing-market benchmark. Review the report before applying.')
        root=Path(__file__).resolve().parent
        if st.button('Load included NFL history',key='nfl_load_included_history'):
            try:st.session_state['nfl_historical_dataset']=json.loads((root/'nfl_history_2016_2025.json').read_text());st.success('Included NFL historical dataset loaded.')
            except (OSError,ValueError) as exc:st.error(str(exc))
        if st.button('Load included NFL fit results',key='nfl_load_included_fit'):
            try:
                payload=json.loads((root/'nfl_reference_fit.json').read_text())
                if payload['model']['core_signature']!=signature(data['parameters']):raise ValueError('NFL core settings differ from the included fit. Train your own fit or restore the initial NFL core settings.')
                st.session_state['nfl_training_result']={**payload,'forecasts':{}};st.success('Reference report loaded; review and apply below.')
            except (OSError,ValueError,TypeError,KeyError) as exc:st.error(str(exc))
    uploads=st.file_uploader('Upload historical NFL files',type=['csv','gz','json','zip'],accept_multiple_files=True,key='nfl_history_upload')
    st.caption('Use games.csv plus play_by_play_YEAR.csv.gz, or restore the compact JSON exported below. Whole-season averages cannot reconstruct earlier pregame inputs.')
    merge=st.checkbox('Merge with NFL history already loaded',value=True,key='nfl_history_merge')
    if st.button('Import and check NFL historical files',disabled=not uploads,key='nfl_import_history'):
        try:
            existing=st.session_state.get('nfl_historical_dataset') if merge else None
            st.session_state['nfl_historical_dataset']=import_files([(u.name,u.getvalue()) for u in uploads],existing)
            st.session_state.pop('nfl_training_result',None)
            st.success('NFL history imported. Review coverage below.')
        except (ValueError,TypeError,RuntimeError,zipfile.BadZipFile) as exc:st.error(str(exc))
    with st.expander('Download NFL history for free',expanded=True):
        years=st.multiselect('NFL historical seasons',list(range(1999,datetime.now().year+1)),default=list(range(2016,2026)),key='nfl_history_years')
        postseason=st.checkbox('Include NFL postseason games',value=False,key='nfl_history_postseason',help='Adds playoffs. Playoff forecasts can use earlier regular-season games and earlier playoff rounds, but never their own results.')
        st.caption('No API key needed. Each season is a sizeable compressed download; ten seasons can take several minutes. Only compact summaries remain in the training dataset. Start with one year if you want to check the feed first.')
        if st.button('Download selected NFL seasons',disabled=not years,key='nfl_fetch_history'):
            bar=st.progress(0.,text='Connecting to nflverse')
            try:
                dataset,errors=fetch_years(years,progress=lambda v,t:bar.progress(v,text=t),existing=st.session_state.get('nfl_historical_dataset') if merge else None,include_postseason=postseason)
                st.session_state['nfl_historical_dataset']=dataset;st.session_state.pop('nfl_training_result',None)
                for error in errors:st.warning(error)
                if not errors:st.success('NFL seasons downloaded. Review coverage and train below.')
            except (ValueError,TypeError,RuntimeError,KeyError) as exc:st.error(str(exc))
            finally:bar.empty()
    dataset=st.session_state.get('nfl_historical_dataset')
    if dataset:
        st.subheader('NFL historical coverage')
        st.dataframe(pd.DataFrame(coverage(dataset)),hide_index=True,width='stretch')
        if dataset['issues']:st.dataframe(pd.DataFrame(dataset['issues']),hide_index=True,width='stretch')
        st.download_button('Export checked NFL historical dataset',json.dumps(dataset,allow_nan=False),file_name='nfl_historical_dataset.json',mime='application/json',key='nfl_export_history')
        st.caption('Save this compact dataset to restore history after a restart. The full model-input export stores applied models and settings, while historical data is saved separately here.')
        a,b=st.columns(2)
        validation=int(a.number_input('First NFL validation season',min_value=2000,max_value=2100,value=None if 'nfl_validation' in st.session_state else 2023,step=1,key='nfl_validation'))
        test=int(b.number_input('First untouched NFL test season',min_value=2001,max_value=2100,value=None if 'nfl_test' in st.session_state else 2025,step=1,key='nfl_test'))
        st.write(f'Train before {validation}; choose model settings using {validation}–{test-1}; evaluate once on {test} onward. First two weeks and incomplete team histories are withheld.')
        st.caption('Inputs: prior-week EPA, success, yards, explosive passes, dropback/rush EPA, completion percentage versus expectation, sack rates, pace, turnovers, opponent-adjusted result ratings, rest days and venue. Closing spreads and totals are evaluation benchmarks, never prediction inputs. Historical injury, weather, ATS and discretionary context adjustments are not trained by these files.')
        if st.button('Build prior-week features and train NFL gradient boosting',key='nfl_train'):
            bar=st.progress(0.,text='Building NFL historical features')
            try:
                fingerprint=dataset_hash(dataset);core=signature(data['parameters']);key=(fingerprint,core)
                cached=st.session_state.get('nfl_training_rows')
                if cached and cached['key']==key:rows,skipped=cached['rows'],cached['skipped']
                else:
                    rows,skipped=build_rows(dataset,data['parameters'],progress=lambda v,t:bar.progress(v,text=t))
                    st.session_state['nfl_training_rows']=dict(key=key,rows=rows,skipped=skipped)
                with st.spinner('Fitting NFL margin and total corrections…'):
                    model,report,forecasts=fit_boost(rows,data['parameters'],validation,test,fingerprint,league='NFL')
                report['withheld_games']=skipped
                st.session_state['nfl_training_result']=dict(model=model,report=report,forecasts=forecasts)
                st.success('NFL fit complete. Review later-season accuracy before applying.')
            except (ValueError,TypeError,KeyError,ImportError) as exc:st.error(str(exc))
            finally:bar.empty()
    else:st.info('NFL history is not loaded yet. Live statistical projections still work; NFL gradient boosting stays inactive until you train and apply a qualifying fit.')
    fitted=st.session_state.get('nfl_training_result')
    if fitted:
        report=fitted['report'];st.subheader('NFL accuracy and calibration')
        rows=[dict(target=t,split=s,games=report['counts'][s],enabled=details['enabled'],baseline_RMSE=m['baseline']['rmse'],selected_RMSE=m['selected']['rmse'],selected_MAE=m['selected']['mae'],bias=m['selected']['bias']) for t,details in report['targets'].items() for s,m in details['metrics'].items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.caption('Smaller MAE and RMSE mean smaller score errors. Positive bias means the model predicted too few points. Only targets improving validation RMSE activate. Test outcomes never choose tree settings or uncertainty.')
        for target,counts in report.get('probability_test_coverage',{}).items():
            withheld=counts['non_half_point_lines']+counts['invalid_lines']
            if withheld:
                st.caption(f"{target.title()} probability checks scored {counts['scored_games']} test games; excluded {withheld} invalid or non-whole/half-point reference lines. {counts['missing_lines']} games had no line. Eligible games still enter training and finite reference lines remain in the point-error comparison.")
        for target,metrics in report.get('probability_test',{}).items():
            st.markdown('**'+target.title()+' cover-probability checks on the untouched test seasons**')
            st.caption(f"Games: {metrics['games']}; three-outcome Brier score: {metrics['three_outcome_brier']:.4f}; log loss: {metrics['log_loss']:.4f}. Lower is better. Reliability compares estimated and observed results.")
            st.dataframe(pd.DataFrame(metrics['reliability']),hide_index=True,width='stretch')
        with st.expander('NFL complete fit report and market comparison'):st.json(report)
        unchanged=signature(data['parameters'])==report['core_signature']
        qualifies=any(t['enabled'] for t in report['targets'].values())
        if st.button('Apply validated NFL model and uncertainty',disabled=not unchanged or not qualifies,key='nfl_apply_model'):
            data['nfl_boost_model']=copy.deepcopy(fitted['model']);data['nfl_training_report']=copy.deepcopy(report)
            for target,key in [('margin','sigma_spread'),('total','sigma_total')]:data['parameters']['NFL'][key]=report['targets'][target]['validation_sigma']
            reset_widgets();rerun()
        if not unchanged:st.caption('NFL core settings changed since training. Restore them or refit before applying.')
        if not qualifies:st.caption('Neither correction beat the statistical baseline on validation; the baseline stays selected.')
        st.download_button('Download trained NFL model',json.dumps({'model':fitted['model'],'report':report},allow_nan=False),file_name='nfl_trained_model.json',mime='application/json',key='nfl_model_download')
        forecasts=[dict(r,features=json.dumps(r['features']),split=split) for split,group in fitted['forecasts'].items() for r in group]
        st.download_button('Download NFL historical backtest rows',pd.DataFrame(forecasts).to_csv(index=False),file_name='nfl_backtest_rows.csv',mime='text/csv',key='nfl_backtest_download')
    with st.expander('Restore a saved NFL model'):
        saved=st.file_uploader('Upload trained NFL model JSON',type=['json'],key='nfl_model_restore')
        if st.button('Restore NFL model',disabled=not saved,key='nfl_restore_apply'):
            try:
                payload=json.loads(saved.getvalue());model=payload['model'];report=payload['report']
                if model.get('core_signature')!=signature(data['parameters']):raise ValueError('Restore the same NFL core settings used when this model was trained.')
                correction(model,{})
                if not any(t['enabled'] for t in model['targets'].values()):raise ValueError('This saved fit did not qualify on validation.')
                uncertainty={}
                for target,key in [('margin','sigma_spread'),('total','sigma_total')]:
                    value=float(report['targets'][target]['validation_sigma'])
                    if not 0<value<100:raise ValueError('Invalid validation uncertainty.')
                    uncertainty[key]=value
                data['nfl_boost_model']=model;data['nfl_training_report']=report
                data['parameters']['NFL'].update(uncertainty)
                reset_widgets();rerun()
            except (ValueError,TypeError,KeyError,IndexError) as exc:st.error(str(exc))
