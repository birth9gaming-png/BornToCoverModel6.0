"""Standalone NFL anytime-TD postprocessor using pandas/numpy.

Only scoring a TD counts: throwing a passing TD does not. Recency uses the
most recent ACTIVE GAME by kickoff, never calendar week minus one.
"""
from __future__ import annotations
from collections import defaultdict
import math
import re
import numpy as np
import pandas as pd
from nfl_features_ext import eligible_plays,numeric,utc
from nfl_stats import code


def name_key(value):
    parts=re.sub(r'[^a-z0-9 ]','',str(value).lower()).split()
    return ' '.join(p for p in parts if p not in ('jr','sr','ii','iii','iv'))


def active_game_history(game_logs,as_of):
    """Bye/inactive rows are dropped BEFORE the chronological last-game lookup.

    Include zero-touch active games in the input. `active=True` should come
    from actual snaps/participation, not roster membership. Missing logs are
    not zero-TD games. Future games cannot leak into any recency calculation.
    """
    df=pd.DataFrame(game_logs).copy()
    if df.empty:return df
    required={'player_id','game_id','kickoff','active','touchdowns'}
    if required-set(df):raise ValueError('Player game logs missing '+', '.join(sorted(required-set(df))))
    df['_at']=df.kickoff.map(utc);cutoff=utc(as_of)
    if pd.isna(cutoff):raise ValueError('ATTD as_of needs an explicit timezone.')
    active=df.active.map(lambda v:isinstance(v,(bool,np.bool_)) and bool(v))
    keep=active & df._at.notna() & df._at.lt(cutoff)
    if 'completed' in df:keep &= df.completed.eq(True)
    df=df[keep].copy();df['touchdowns']=pd.to_numeric(df.touchdowns,errors='coerce')
    if df.touchdowns.isna().any() or df.touchdowns.lt(0).any() or df.touchdowns.mod(1).ne(0).any():raise ValueError('Actual player TD counts must be known nonnegative integers.')
    if df.duplicated(['player_id','game_id']).any():raise ValueError('Player logs contain duplicate player/game rows.')
    return df.sort_values(['player_id','_at','game_id'],ascending=[True,False,False])


def last_active_games(game_logs,as_of):
    """Example: week 4 active → week 5 bye → week 6 forecast selects week 4."""
    df=active_game_history(game_logs,as_of)
    return df.groupby('player_id',sort=False).head(1).drop(columns='_at') if not df.empty else df


def _number(value):
    try:value=float(value)
    except (ValueError,TypeError):return None
    return value if math.isfinite(value) else None


def rate_anytime_touchdowns(players,game_logs,games,config,as_of,opponent_metrics=None,quotes=None,injury_reports=None,auto_injury_reports=None,availability_settings=None):
    """Filter market total first, then combine usage, xTD, matchup and recency.

    Returns scenario probabilities and price-based EV only when a current
    American-odds ATTD quote is available. Weights are editable, unfitted priors.
    `xTD` must be expected scorer TDs; passing TDs are not scorer TDs.
    """
    if not config['enabled'] or not config['attd_enabled']:return pd.DataFrame()
    players=pd.DataFrame(players);games=pd.DataFrame(games);quotes=pd.DataFrame(quotes if quotes is not None else [])
    if players.empty or games.empty:return pd.DataFrame()
    if not {'id','away','home','total'}.issubset(games):raise ValueError('ATTD games need id, away, home and market total.')
    if not {'player_id','team','position'}.issubset(players):raise ValueError('ATTD players need player_id, team and position.')
    if 'league' in games:games=games[games.league.eq('NFL')]
    pool=games[pd.to_numeric(games.total,errors='coerce').ge(config['attd_min_total'])].copy()
    pool_by_team=defaultdict(list)
    for _,g in pool.iterrows():
        for side in ('away','home'):pool_by_team[code(g[side])].append(g.to_dict())
    # A date window can include two games for the same team. Rate both pools.
    expanded=[]
    for _,player in players.iterrows():
        for game in pool_by_team.get(code(player.team),[]):expanded.append((player,game))
    now=utc(as_of)
    history=active_game_history(game_logs,as_of)
    output=[];opponent_metrics=opponent_metrics or {}
    # Resolve each report once. The previous per-candidate roster scan was
    # quadratic in players and repeated even for unrelated CFB reports.
    # Keep the unique-name guard, including ambiguous first-initial matches.
    from auto_injuries import same_player,_recent
    roster_by_name=defaultdict(list)
    for p in players.to_dict('records'):
        parts=name_key(p.get('name','')).split()
        if parts:
            roster_by_name[(code(p.get('team','')),parts[-1],parts[0][0])].append(p)
    automatic_by_player=defaultdict(list)
    for report in auto_injury_reports or []:
        team=code(report.get('team_id',''))
        if report.get('league')!='NFL' or team not in pool_by_team:continue
        parts=name_key(report.get('player','')).split()
        if not parts:continue
        matches=[p for p in roster_by_name.get((team,parts[-1],parts[0][0]),[])
                 if same_player(report.get('player',''),p.get('name',''))]
        if len(matches)==1:
            automatic_by_player[(team,name_key(matches[0].get('name','')))].append(report)
    for player,g in expanded:
        team=code(player.team)
        position=str(player.position).upper()
        rank=_number(player.get('depth_rank',player.get('usage_rank')))
        roles={'RB':2,'WR':2,'TE':1}
        if position not in roles or rank is None or rank>roles[position] or rank<1:continue
        kickoff=utc(g.get('kickoff'));cutoff=min(now,kickoff) if pd.notna(kickoff) else now
        status=str(player.get('status','')).upper()
        # Free feed rows are observations, not reviewed manual point estimates.
        # They must precede the forecast and kickoff and be less than a day old.
        automatic=[]
        for report in automatic_by_player.get((team,name_key(player.get('name',''))),[]):
            at=utc(report.get('observed_at'))
            if pd.notna(at) and at<=cutoff and cutoff-at<pd.Timedelta(hours=24) and _recent(report,cutoff.to_pydatetime(),(availability_settings or {}).get('max_report_days',14)):automatic.append(report)
        if automatic:
            latest=max(automatic,key=lambda v:str(v.get('report_updated_at','')))
            if latest.get('status')!='Unknown':status=str(latest['status']).upper()
        reports=[]
        for report in injury_reports if injury_reports is not None else []:
            published=utc(report.get('report_as_of'))
            if report.get('reviewed') and report.get('league')=='NFL' and code(report.get('team_id',''))==team and name_key(report.get('player'))==name_key(player.get('name')) and pd.notna(published) and published<=cutoff and pd.notna(kickoff) and kickoff-published<=pd.Timedelta(hours=96):reports.append(report)
        if reports:
            latest=max(reports,key=lambda v:utc(v['report_as_of']));status=str(latest['status']).upper()
        if status in ('OUT','INACTIVE','IR','PUP','SUSPENDED','DOUBTFUL','INA','RES'):continue
        logs=history[history.player_id.eq(player.player_id) & history._at.lt(cutoff)].head(int(config['attd_window'])) if not history.empty else history
        if logs.empty:continue
        volume=(numeric(logs,'carries',0).fillna(0)+numeric(logs,'targets',0).fillna(0)).mean()
        supplied_volume=_number(player.get('projected_opportunities'))
        volume=supplied_volume if supplied_volume is not None else float(volume)
        minimum=config['attd_min_'+position.lower()+'_volume']
        if volume<minimum:continue
        last=logs.iloc[0];last_td=int(last.touchdowns)
        last_known=bool(player.get('recency_verified',True))
        # A zero-touch active row is still the most recent game and still has 0 TDs.
        xtd=numeric(logs,'xTD');expected=_number(player.get('projected_xTD'))
        if expected is None and xtd.notna().all():expected=float(xtd.mean())
        if expected is None or expected<0:continue
        lam=expected;actual=float(logs.touchdowns.sum());xtd_sum=float(xtd.sum()) if xtd.notna().all() else None
        regression=0.
        if config['attd_xtd_enabled'] and xtd_sum is not None:
            regression=config['attd_xtd_weight']*max(0.,(xtd_sum-actual)/len(logs));lam+=regression
        def share(num,den):
            n,d=numeric(logs,num).sum(min_count=1),numeric(logs,den).sum(min_count=1)
            return float(n/d) if pd.notna(n) and pd.notna(d) and d>0 else None
        target_share=share('redzone_targets','team_redzone_targets')
        touch_share=share('redzone_touches','team_redzone_touches')
        rz=touch_share if position=='RB' else target_share
        if config['attd_redzone_enabled'] and rz is not None:lam*=1+config['attd_redzone_weight']*float(np.clip(rz-.15,-.15,.65))
        opponent=code(g['home'] if team==code(g['away']) else g['away'])
        rz_dvoa=_number(opponent_metrics.get(opponent,{}).get('redzone_defensive_dvoa'))
        if config['attd_defense_enabled'] and rz_dvoa is not None:lam*=1+config['attd_defense_weight']*float(np.clip(rz_dvoa/100,-.5,.5))
        baseline=1-float(np.exp(-expected));probability=1-float(np.exp(-max(0.,lam)))
        rule='No recency adjustment'
        if config['attd_recency_enabled'] and last_known:
            if last_td==0:probability+=config['attd_bounce_bonus'];rule='Bounce-back: 0 TD in last active game'
            elif last_td>=2:probability*=config['attd_sell_multiplier'];rule='Sell high: 2+ TD in last active game'
        elif not last_known:rule='Active-game coverage incomplete; recency skipped'
        probability=float(np.clip(probability,0.,config['attd_probability_cap']))
        if_active=probability
        availability=1-float((availability_settings or {}).get('questionable_absence',.25)) if status=='QUESTIONABLE' else 1.
        probability*=availability
        row=dict(game_id=str(g['id']),player_id=str(player.player_id),player=player.get('name',str(player.player_id)),team=team,opponent=opponent,
            position=position,status=status,depth_rank=int(rank),role_source=player.get('role_source','supplied depth rank'),market_total=float(g['total']),
            prior_active_game=str(last.game_id),prior_active_week=last.get('week'),prior_active_kickoff=last.kickoff,
            prior_active_tds=last_td,active_games_used=len(logs),opportunities_per_game=volume,
            redzone_target_share=target_share,redzone_touch_share=touch_share,xTD_per_game=expected,
            xTD_window=xtd_sum,actual_tds_window=actual,xTD_regression_lambda=regression,
            xTD_source=player.get('xTD_source',last.get('xTD_source','supplied xTD')),
            opponent_redzone_dvoa=rz_dvoa,baseline_probability=baseline,attd_probability=probability,
            recency_rule=rule,availability_probability=availability,attd_if_active_probability=if_active,
            book=None,odds=None,quote_as_of=None,conditional_ev=None,availability_weighted_ev=None,implied_probability=None)
        valid=[]
        if not quotes.empty:
            for _,quote in quotes.iterrows():
                if str(quote.get('game_id'))!=str(g['id']) or str(quote.get('player_id'))!=str(player.player_id):continue
                price=_number(quote.get('odds'));at=utc(quote.get('quoted_at'))
                if price is None or abs(price)<100 or pd.isna(at) or at>cutoff or cutoff-at>pd.Timedelta(hours=6):continue
                profit=100/abs(price) if price<0 else price/100
                ev=if_active*profit-(1-if_active)
                valid.append(dict(book=quote.get('book'),odds=price,quote_as_of=quote.get('quoted_at'),conditional_ev=ev,availability_weighted_ev=availability*ev,
                                  implied_probability=abs(price)/(abs(price)+100) if price<0 else 100/(price+100)))
        if valid:row.update(max(valid,key=lambda v:v['conditional_ev']))
        output.append(row)
    result=pd.DataFrame(output)
    if not result.empty:result=result.sort_values(['conditional_ev','attd_probability'],ascending=False,na_position='last')
    return result


def build_player_inputs(pbp,schedule,roster,snaps,through_week):
    """Automatic prior-game logs from PBP + actual snaps, including 0-touch games.

    xTD is a labeled proxy: smoothed league TD frequencies by opportunity type
    and field-position bucket, fit using only the eligible prior weeks.
    Roles are ranked by recent usage unless the user supplies depth-chart ranks.
    """
    plays=eligible_plays(pbp,through_week);schedule=pd.DataFrame(schedule).copy();roster=roster.copy();snaps=snaps.copy()
    if plays.empty or roster.empty:return [],[]
    if not {'gsis_id','team','position'}.issubset(roster):raise ValueError('NFL roster needs gsis_id, team and position.')
    roster=roster[roster.gsis_id.notna()].drop_duplicates('gsis_id',keep='last')
    roster['team']=roster.team.map(code);people=roster.set_index('gsis_id').to_dict('index')
    if 'week' in schedule:schedule=schedule[numeric(schedule,'week').le(through_week)]
    if 'away_score' in schedule:schedule=schedule[schedule.away_score.notna() & schedule.home_score.notna()]
    records={}
    for _,game in schedule.iterrows():
        gid=str(game.get('game_id',game.get('id')));kickoff=game.get('kickoff')
        if not kickoff:kickoff=str(game.get('gameday'))+' '+str(game.get('gametime'))+' America/New_York'
        records[gid]=dict(game_id=gid,kickoff=kickoff,week=int(game['week']),season=int(game['season']))
    opportunities=[]
    for kind,column,mask in [('rush','rusher_player_id',plays.designed_rush),('target','receiver_player_id',plays.dropback)]:
        if column not in plays:continue
        selected=plays[mask & plays[column].notna()].copy()
        if selected.empty:continue
        selected['player_id']=selected[column];selected['kind']=kind
        selected['bucket']=pd.cut(numeric(selected,'yardline_100'),[-1,3,10,20,50,101],labels=False).fillna(4).astype(int)
        selected['scored']=numeric(selected,'rush_touchdown' if kind=='rush' else 'pass_touchdown',0).eq(1).astype(int)
        selected['redzone']=numeric(selected,'yardline_100').le(20).astype(int)
        selected['touch']=1 if kind=='rush' else numeric(selected,'complete_pass',0).eq(1).astype(int)
        opportunities.append(selected[['game_id','posteam','player_id','kind','bucket','scored','redzone','touch']])
    if not opportunities:return [],[]
    opps=pd.concat(opportunities,ignore_index=True)
    rates=opps.groupby(['kind','bucket']).scored.agg(['sum','count'])
    league=opps.groupby('kind').scored.mean()
    rates['rate']=(rates['sum']+20*rates.index.get_level_values(0).map(league))/(rates['count']+20)
    opps['xTD']=[float(rates.loc[(k,b),'rate']) for k,b in zip(opps.kind,opps.bucket)]
    team_rz=opps.groupby(['game_id','posteam']).apply(lambda f:pd.Series(dict(team_redzone_targets=int((f.kind.eq('target') & f.redzone.eq(1)).sum()),team_redzone_touches=int((f.redzone*f.touch).sum()))),include_groups=False).to_dict('index')
    logs={}
    def ensure(gid,pid,team):
        if gid not in records:return None
        key=(gid,pid)
        if key not in logs:logs[key]=dict(**records[gid],player_id=pid,team=team,active=True,completed=True,touchdowns=0,carries=0,targets=0,redzone_targets=0,redzone_touches=0,xTD=0.,xTD_source='PBP field-position frequency proxy',**team_rz.get((gid,team),{}))
        return logs[key]
    for (gid,pid),group in opps.groupby(['game_id','player_id']):
        row=ensure(str(gid),str(pid),group.iloc[0].posteam)
        if row is None:continue
        row.update(carries=int(group.kind.eq('rush').sum()),targets=int(group.kind.eq('target').sum()),xTD=float(group.xTD.sum()),redzone_targets=int((group.kind.eq('target') & group.redzone.eq(1)).sum()),redzone_touches=int((group.redzone*group.touch).sum()))
    # PFR snaps supply actual active games even with 0 targets/carries, and skip
    # genuine inactive games. Team bye weeks have no snap/game row at all.
    pfr_to_gsis={str(v['pfr_id']):pid for pid,v in people.items() if pd.notna(v.get('pfr_id'))}
    snap_games=set()
    if {'game_id','pfr_player_id','team','offense_snaps'}.issubset(snaps):
        snaps=snaps[numeric(snaps,'week').le(through_week)].copy()
        snap_games=set((str(g),code(t)) for g,t in zip(snaps.game_id,snaps.team))
        active=numeric(snaps,'offense_snaps',0).fillna(0)+numeric(snaps,'st_snaps',0).fillna(0)>0
        for _,s in snaps[active].iterrows():
            pid=pfr_to_gsis.get(str(s.pfr_player_id))
            if pid:ensure(str(s.game_id),pid,code(s.team))
    # td_player_id identifies the scorer, not the passer; include return TDs.
    if {'td_player_id','touchdown','week'}.issubset(pbp):
        scoring=pbp[numeric(pbp,'week').le(through_week) & numeric(pbp,'touchdown',0).eq(1) & numeric(pbp,'no_play',0).fillna(0).eq(0) & pbp.td_player_id.notna()]
        for (gid,pid),count in scoring.groupby(['game_id','td_player_id']).size().items():
            if pid in people:
                row=ensure(str(gid),str(pid),code(people[pid]['team']))
                if row is not None:row['touchdowns']=int(count)
    else:
        for (gid,pid),group in opps.groupby(['game_id','player_id']):
            row=logs.get((str(gid),str(pid)))
            if row is not None:row['touchdowns']=int(group.scored.sum())
    frame=pd.DataFrame(logs.values())
    if frame.empty:return [],[]
    frame['_at']=frame.kickoff.map(utc);candidates=[]
    for pid,info in people.items():
        if info['position'] not in ('RB','WR','TE'):continue
        prior=frame[frame.player_id.eq(pid)].sort_values('_at',ascending=False).head(4)
        if prior.empty:continue
        team=code(info['team']);current=prior[prior.team.eq(team)]
        if current.empty:continue
        expected_games=[gid for gid in records if ('_'+team+'_' in gid or gid.endswith('_'+team))]
        verified=bool(pd.notna(info.get('pfr_id')) and expected_games and all((gid,team) in snap_games for gid in expected_games))
        candidates.append(dict(player_id=str(pid),name=info.get('full_name',info.get('name',str(pid))),team=team,position=info['position'],status=info.get('status',''),
            projected_opportunities=float((current.carries+current.targets).mean()),recency_verified=verified,role_source='ranked by prior active-game usage',xTD_source='PBP field-position frequency proxy'))
    candidate=pd.DataFrame(candidates)
    if not candidate.empty:
        candidate=candidate.sort_values(['projected_opportunities','player_id'],ascending=[False,True]);candidate['usage_rank']=candidate.groupby(['team','position']).cumcount()+1
    return frame.drop(columns='_at').to_dict('records'),candidate.to_dict('records')
