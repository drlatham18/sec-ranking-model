"""Six-hour baseline; half-hour refresh on SEC game dates in Eastern time."""
import json
import os
import pathlib
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from best_bets import get, stamp


def due(model, now, event='schedule', scoreboard=None):
    if event != 'schedule' or (now.hour % 6 == 0 and now.minute < 30):
        return True
    today = now.astimezone(ZoneInfo('America/New_York')).date()
    for game in model['games']:
        kickoff = stamp(game.get('kickoff'))
        day = kickoff.astimezone(ZoneInfo('America/New_York')).date() if kickoff else game.get('date')
        if str(day) == str(today):
            return True
    # Detect rescheduled games without relying on the committed schedule.
    teams = {t['team'] for t in model['teams']}
    return any(c.get('team', {}).get('location') in teams
               for e in (scoreboard or {}).get('events', [])
               for competition in e.get('competitions', [])
               for c in competition.get('competitors', []))


if __name__ == '__main__':
    now = datetime.now(timezone.utc)
    model = json.loads(pathlib.Path('output/app_data.json').read_text())
    event = os.getenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    run = due(model, now, event)
    if not run:
        try:
            today = now.astimezone(ZoneInfo('America/New_York')).strftime('%Y%m%d')
            board = get('https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard',
                        dates=today, groups=80, limit=1000)
            run = due(model, now, event, board)
        except Exception:
            run = True  # Source outage must not silently suppress a game-day refresh.
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write('due=' + str(run).lower() + '\n')
