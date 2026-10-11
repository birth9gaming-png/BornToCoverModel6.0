"""Explicit user preferences; these are editable hypotheses, not fitted facts."""
from __future__ import annotations

PROFILE=dict(offense_share=.65,shrink=1.,by=3.,bs=.55,fei_weight=.40,
    lookahead_margin_points=-2.,big_win_margin_points=-2.,context_cap_points=3.,
    context_ranked_only=True,selection_style='fade_spots')


def apply_profile(data):
    data['parameters']['CFB'].update(PROFILE)
    data['cfb_preference_profile']='Offensive efficiency and ranked-opponent situations'
    return data


def flags(result,side):
    info=result.get('schedule_context',{}).get(side,{})
    return [label for key,label in [('lookahead','ranked-opponent look-ahead'),('big_win','after a major win')]
            if info.get(key) is True]


def focus_signal(result,quote):
    if quote['kind']!='spread':return False
    opposite='home' if quote['selection']=='away' else 'away'
    return bool(flags(result,opposite)) and not flags(result,quote['selection'])
