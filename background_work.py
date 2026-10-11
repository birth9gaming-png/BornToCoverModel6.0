"""Small, detached feed jobs. Worker threads never access Streamlit state."""
from __future__ import annotations
from concurrent.futures import Future
from contextvars import ContextVar
from threading import Event,Lock,Thread
import time

_REPORTER=ContextVar('football_feed_reporter',default=None)


class RefreshCancelled(RuntimeError):pass


def report_progress(message):
    callback=_REPORTER.get()
    if callback:callback(message)


class FeedJob:
    def __init__(self,work,source,kind='automatic',deadline=600):
        self.source=source;self.kind=kind;self.started=time.monotonic()
        self.future=Future();self._cancel=Event();self._lock=Lock()
        self._message='Starting data update';self.deadline=deadline
        def target():
            token=_REPORTER.set(self.progress)
            try:
                result=work()
                self.progress('Data update complete')
                self.future.set_result(result)
            except Exception as exc:self.future.set_exception(exc)
            finally:_REPORTER.reset(token)
        self.thread=Thread(target=target,name='football-feed-update',daemon=True)
        self.thread.start()

    def progress(self,message):
        if self._cancel.is_set():raise RefreshCancelled('Refresh stopped; saved inputs retained.')
        if time.monotonic()-self.started>self.deadline:
            raise RefreshCancelled('Refresh exceeded ten minutes; saved inputs retained. Check the last feed shown and try again.')
        with self._lock:self._message=str(message)

    def status(self):
        with self._lock:return self._message,int(time.monotonic()-self.started)

    def cancel(self):self._cancel.set()


def request_label(url,params=None):
    """Human progress labels never include query strings or credentials."""
    from urllib.parse import urlparse
    parsed=urlparse(url);path=parsed.path;year=(params or {}).get('year')
    if 'play_by_play' in path:return 'Loading NFL play-by-play statistics'
    if '/games.csv' in path:return 'Loading NFL schedules and results'
    if 'snap_counts' in path:return 'Loading NFL player snap counts'
    if 'roster_' in path:return 'Loading NFL player rosters'
    if 'nflverse' in url:return 'Loading NFL charting inputs'
    if 'open-meteo' in parsed.netloc:return 'Checking NFL kickoff wind forecasts'
    if 'the-odds-api' in parsed.netloc:return 'Loading current bookmaker odds'
    if 'collegefootballdata' in parsed.netloc:
        return 'Loading CFB '+path.strip('/').replace('/',' / ')+(f' ({year})' if year else '')
    if 'bcftoys' in parsed.netloc:return 'Loading CFB FEI efficiency ratings'
    if 'espn' in parsed.netloc or 'covers.com' in parsed.netloc:return 'Checking current injury reports'
    return 'Loading a football data feed'
