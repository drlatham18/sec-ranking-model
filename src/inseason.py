"""In-season strength adjustment: blend frozen preseason ratings with results to date.

The preseason model is opponent-adjusted and validated, but it cannot see the
season being played. This module folds completed games back into each team's
rating so a matchup uses current strength rather than an August forecast.

Method (all constants estimated, none assigned by hand -- see fit_inseason.py):

1. Every completed game against a *rated* opponent yields an implied rating for
   each side, inverting the fitted game model:

       implied(team) = rating(opponent) + (own_margin - hfa * home_flag) / b1

   This is opponent- and venue-adjusted, and lands on the same points scale as
   the preseason rating, so the two are directly blendable.

2. A team's in-season rating is the mean of its implied ratings.

3. Preseason and in-season are combined by shrinkage, so early noisy results
   move a rating only a little and a full season moves it a lot:

       weight = n / (n + K)
       blended = (1 - weight) * preseason + weight * in_season

   K is chosen by leave-one-season-out log loss on development seasons only.

Games against unrated (FCS) opponents carry no opponent rating and are skipped:
a scoreless anchor would bias every team that played one.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "output" / "inseason_calibration.json"


def load_calibration(path=None):
    """Fitted blend constants. Raises rather than guessing if fitting never ran."""
    path = pathlib.Path(path or CALIBRATION)
    if not path.exists():
        raise FileNotFoundError(
            "%s missing -- run: python src/fit_inseason.py" % path)
    return json.loads(path.read_text())


def implied_ratings(games, ratings, b1, hfa):
    """Per-team implied ratings from completed games against rated opponents.

    games: iterable of dicts with home, away, home_points, away_points, neutral.
    Returns {team: [implied, ...]}.
    """
    out = {}
    for g in games:
        home, away = g["home"], g["away"]
        if home not in ratings or away not in ratings:
            continue
        if g.get("home_points") is None or g.get("away_points") is None:
            continue
        margin = g["home_points"] - g["away_points"]
        home_flag = 0.0 if g.get("neutral") else 1.0
        for team, opponent, sign, flag in ((home, away, 1, home_flag),
                                           (away, home, -1, -home_flag)):
            implied = ratings[opponent] + (sign * margin - hfa * flag) / b1
            out.setdefault(team, []).append(implied)
    return out


def blend(ratings, implied, k):
    """Shrink each preseason rating toward its in-season evidence.

    Returns {team: {"preseason", "in_season", "blended", "n_games", "weight"}}
    for every rated team, including those that have not played a rated opponent.
    """
    if k <= 0:
        raise ValueError("shrinkage constant K must be positive")
    out = {}
    for team, preseason in ratings.items():
        values = implied.get(team, [])
        n = len(values)
        weight = n / (n + k)
        season = sum(values) / n if n else None
        out[team] = {
            "preseason": preseason,
            "in_season": season,
            "blended": preseason if not n else (1 - weight) * preseason + weight * season,
            "n_games": n,
            "weight": round(weight, 4),
        }
    return out


def current_ratings(games, ratings, calibration=None):
    """Blended rating per team from the season's completed games."""
    c = calibration or load_calibration()
    implied = implied_ratings(games, ratings, c["scale_b1"], c["scale_hfa"])
    return blend(ratings, implied, c["K"])
