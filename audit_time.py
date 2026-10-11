"""Require explicit timezones for imported observations and quote audits."""
from datetime import datetime
from matchup_history import timestamp


def observed_time(value):
    if not value:raise ValueError('Timestamp with timezone is required.')
    text=str(value)
    if text.endswith(' America/New_York'):return timestamp(text)
    parsed=datetime.fromisoformat(text.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('Timestamp must include a timezone (Z or an explicit offset).')
    return timestamp(text)
