"""ATS and total trends from actual historical games."""
from datetime import datetime
import pandas as pd
import streamlit as st
from trend_analysis import records,scan


def panel(data,league):
    with st.expander(league+' ATS and total trend finder'):
        historical=st.session_state.get('cfb_historical_dataset' if league=='CFB' else 'nfl_historical_dataset')
        rows=records(data,league,historical)
        teams=sorted({g[s] for g in rows for s in ('away','home')})
        if not teams:
            st.caption('Load historical games and betting lines in '+('CFB' if league=='CFB' else 'NFL')+' Historical Training, or refresh the connected slate to load ATS history.')
            return
        if league=='CFB':
            polls=(historical or {}).get('polls',[])+data.get('poll_history',[])
            years=sorted({p['season'] for p in polls})
            st.caption('Historical AP poll seasons loaded: '+(', '.join(map(str,years)) if years else 'none; ranked-opponent trends need AP history')+'. Advanced game rows loaded: '+str(len((historical or {}).get('advanced',[])))+'. The automatic CFB history downloader includes AP polls.')
        a,b,c=st.columns(3)
        team=a.selectbox('Trend team',teams,format_func=lambda tid:data['teams'].get(tid,{}).get('name',tid.removeprefix(league+':')),key='trend_team_'+league)
        start=int(b.number_input('First trend season',min_value=1999,max_value=2100,value=2016,step=1,key='trend_start_'+league))
        minimum=int(c.number_input('Minimum older sample',min_value=10,max_value=200,value=20,step=5,key='trend_min_'+league))
        price=float(st.number_input('Assumed historical American odds',value=-110.,step=5.,key='trend_price_'+league,help='Actual historical prices are not available in these files. This price is used only to calculate illustrative returns and break-even rates.'))
        market=st.selectbox('Trend market',['Both','ATS','Total'],key='trend_market_'+league)
        cutoff=data.get('as_of')
        try:
            table,info=scan(data,league,team,historical,as_of=cutoff,min_decisions=minimum,start_season=start,assumed_odds=price)
            if market!='Both':table=[r for r in table if r['market']==market]
            table.sort(key=lambda r:(r['status']=='Later-game support',r['later_ROI_at_assumed_price'] if r.get('later_ROI_at_assumed_price') is not None else -100),reverse=True)
            st.caption(f"{info['eligible_games']} eligible games. For each fixed pattern, the oldest 70% choose back/fade or over/under; the newest 30% test that choice. All outcomes precede {info['as_of']}.")
            display=pd.DataFrame(table)[['pattern','market','selected_direction','games','ATS_WLP','over_WLP','under_WLP','older_record','later_record','later_win_rate','later_ROI_at_assumed_price','missing_lines','status']]
            for col in ('later_win_rate','later_ROI_at_assumed_price'):display[col]=100*display[col]
            display=display.rename(columns={'pattern':'Situation','market':'Market','selected_direction':'Direction chosen from older games','games':'Games','ATS_WLP':'Full ATS W-L-P','over_WLP':'Full Over W-L-P','under_WLP':'Full Under W-L-P','older_record':'Older selected W-L-P','later_record':'Later selected W-L-P','later_win_rate':'Later win %','later_ROI_at_assumed_price':'Later return % at assumed odds','missing_lines':'Missing lines','status':'Evidence'})
            st.dataframe(display,hide_index=True,width='stretch')
            st.download_button('Download '+league+' ATS and total trend results',pd.DataFrame(table).to_csv(index=False),file_name=league.lower()+'_trend_results.csv',mime='text/csv',key='trend_export_'+league)
            st.caption('Later-game support requires adequate samples and a conservative confidence bound above break-even after allowing for multiple patterns. Returns use the assumed price, not verified historical odds. Ranked-opponent and offensive-efficiency screens require their historical inputs. Trend tables do not automatically change predictions.')
        except (ValueError,TypeError,KeyError) as exc:st.error(str(exc))
