"""Read the SEC's public report feed; no authentication or guessed player values.

The endpoint is used by the public embed linked at secsports.com/fbreports.
Only the report's game date, teams, publication time and player status are kept.
A failed or changed feed yields unavailable, never a healthy-roster assumption.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests

SOURCE = 'https://www.secsports.com/fbreports'
ENDPOINT = 'https://app.hdintelligence.com/api/get-publish-public'


def parse_reports(payload):
    if not isinstance(payload, dict):
        raise ValueError('Unexpected availability feed shape')
    reports = []
    for value in payload.values():
        sides = value.get('games', [])
        if len(sides) != 2:
            raise ValueError('Availability report needs two teams')
        day = value['footer']['date']
        datetime.strptime(day, '%Y-%m-%d')
        posted = datetime.fromisoformat(value['publishDate'] + 'T' + value['postedTime'])
        posted = posted.replace(tzinfo=ZoneInfo('America/Chicago')).astimezone(timezone.utc)
        teams = []
        for side in sides:
            name = side.get('teamDisplayName') or side['teamName']
            rows = side['rows']
            if not isinstance(rows, list):
                raise ValueError('Missing player status rows')
            players = [{'name': str(r['name']), 'status': str(r['status'])}
                       for r in rows if r.get('status') != 'Exempt']
            teams.append({'team': name, 'players': players})
        reports.append({'date': day, 'published_at': posted.isoformat(),
                        'report_type': value.get('ReportType'), 'teams': teams})
    return reports


def fetch_reports():
    result = {'source': SOURCE, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'status': 'unavailable', 'reports': [], 'adjustment_applied': False}
    try:
        response = requests.post(ENDPOINT, json={'sport': 'Football', 'organization': 'SEC',
                                               'conference': 'SEC'}, timeout=30)
        response.raise_for_status()
        result['reports'] = parse_reports(response.json())
        result['status'] = 'available' if result['reports'] else 'no_reports'
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        result['reason'] = 'Official availability feed could not be verified: ' + type(error).__name__
    return result
