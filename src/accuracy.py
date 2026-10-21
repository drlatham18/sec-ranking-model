"""Week-by-week accuracy of the live model, scored as if each week were unseen.

For every week W with final results, ratings are rebuilt from games *before* W
only, then used to predict week W. That is exactly what the model knew at
kickoff, so the score is honest even though the published page later overwrites
each game's probability with newer ratings.

Both the in-season blend and the frozen preseason ratings are scored on the same
games, so the value added by results is visible week to week.

Every week carries data flags instead of silently shrinking: games against
unrated (FCS) teams, unfinished games, rated teams with no game found (a bye or a
missing feed row), and whether the scores were cross-checked by a second source.
No extra API request is made; this reuses the schedule the export already holds.
"""
import math

from scipy.stats import norm

import inseason

TIERS = (0.5, 0.65, 0.75, 0.8, 0.9)


def _metrics(rows, p_key, m_key):
    if not rows:
        return None
    n = len(rows)
    correct = sum((r[p_key] > 0.5) == r["home_won"] for r in rows)
    brier = sum((r[p_key] - r["home_won"]) ** 2 for r in rows) / n
    eps = 1e-6
    ll = -sum(math.log(min(max(r[p_key] if r["home_won"] else 1 - r[p_key], eps), 1))
              for r in rows) / n
    mae = sum(abs(r[m_key] - r["margin"]) for r in rows) / n
    return {"n": n, "correct": int(correct), "accuracy": round(correct / n, 4),
            "brier": round(brier, 4), "log_loss": round(ll, 4),
            "margin_mae": round(mae, 2)}


def _tiers(rows):
    out = []
    for t in TIERS:
        sub = [r for r in rows if max(r["p"], 1 - r["p"]) >= t]
        hits = sum((r["p"] > 0.5) == r["home_won"] for r in sub)
        out.append({"min_confidence": t, "n": len(sub), "correct": int(hits),
                    "accuracy": round(hits / len(sub), 4) if sub else None})
    return out


def _is_final(g):
    return bool(g.get("completed") and g.get("week") is not None
                and g.get("home_points") is not None and g.get("away_points") is not None)


def _cross_check_flag(source_info):
    audit = (source_info or {}).get("audit") or {}
    provider = (source_info or {}).get("provider")
    if not audit.get("performed"):
        return "Scores from %s only; not cross-checked against a second feed." % provider
    if audit.get("disagreed"):
        return ("%d score conflict(s) between feeds; those games were excluded."
                % audit["disagreed"])
    return None


def weekly(games, preseason, calibration, prior_cal, conference_of=None,
           conference=None, source_info=None, min_week=None):
    """Score every completed week from min_week on.

    games: dicts with week, home, away, home_points, away_points, neutral,
           completed (the export's full-season frame as records).
    preseason: {team: frozen preseason rating}.
    calibration: in-season calibration (K, scale_b1, scale_hfa, rating_diff_coef,
                 home_field_advantage, residual_sd).
    prior_cal: preseason game calibration (intercept, rating_diff_coef,
               home_field_advantage, residual_sd, neutral_residual_sd).
    """
    min_week = int(min_week or calibration.get("min_week_validated") or 1)
    final = [g for g in games if _is_final(g)]
    weeks = sorted({int(g["week"]) for g in final if int(g["week"]) >= min_week})
    cross_flag = _cross_check_flag(source_info)
    conference_of = conference_of or {}
    out = []
    for w in weeks:
        before = [g for g in final if int(g["week"]) < w]
        blended = {t: v["blended"] for t, v in
                   inseason.current_ratings(before, preseason, calibration).items()}
        slate = [g for g in games if g.get("week") is not None and int(g["week"]) == w]
        rows, unrated, unfinished = [], [], []
        for g in slate:
            h, a = g["home"], g["away"]
            label = "%s @ %s" % (a, h)
            if not _is_final(g):
                unfinished.append(label)
                continue
            if h not in preseason or a not in preseason:
                unrated.append(label)
                continue
            margin = g["home_points"] - g["away_points"]
            if margin == 0:
                continue
            hf = 0.0 if g.get("neutral") else 1.0
            m = (calibration["rating_diff_coef"] * (blended[h] - blended[a])
                 + calibration["home_field_advantage"] * hf)
            p = float(norm.cdf(m / calibration["residual_sd"]))
            mp = (prior_cal.get("intercept", 0.0)
                  + prior_cal["rating_diff_coef"] * (preseason[h] - preseason[a])
                  + prior_cal["home_field_advantage"] * hf)
            sdp = prior_cal["neutral_residual_sd" if g.get("neutral") else "residual_sd"]
            rows.append({"home": h, "away": a, "home_points": int(g["home_points"]),
                         "away_points": int(g["away_points"]), "neutral": bool(g.get("neutral")),
                         "margin": margin, "home_won": margin > 0,
                         "m": m, "p": p, "mp": mp, "pp": float(norm.cdf(mp / sdp))})
        seen = {t for g in slate for t in (g["home"], g["away"])}
        idle = sorted(t for t in preseason if t not in seen)
        conf_rows = [r for r in rows if conference and conference in
                     (conference_of.get(r["home"]), conference_of.get(r["away"]))]
        flags = []
        if cross_flag:
            flags.append(cross_flag)
        if unfinished:
            flags.append("Partial week: %d game(s) not final yet." % len(unfinished))
        if unrated:
            flags.append("%d game(s) involve an unrated (FCS) team and are not scored."
                         % len(unrated))
        if idle:
            flags.append("%d rated team(s) have no game in the feed (bye, or missing data)."
                         % len(idle))
        if not rows:
            flags.append("No scorable games this week.")
        misses = sorted((r for r in rows if (r["p"] > 0.5) != r["home_won"]),
                        key=lambda r: -max(r["p"], 1 - r["p"]))
        out.append({
            "week": w,
            "complete": not unfinished,
            "model": _metrics(rows, "p", "m"),
            "preseason_only": _metrics(rows, "pp", "mp"),
            "conference": ({"name": conference, "model": _metrics(conf_rows, "p", "m"),
                            "preseason_only": _metrics(conf_rows, "pp", "mp")}
                           if conference else None),
            "tiers": _tiers(rows),
            "misses": [{"home": r["home"], "away": r["away"],
                        "home_points": r["home_points"], "away_points": r["away_points"],
                        "favorite": r["home"] if r["p"] > 0.5 else r["away"],
                        "confidence": round(max(r["p"], 1 - r["p"]), 4)}
                       for r in misses],
            "flags": flags,
            "unrated_games": unrated,
            "unfinished_games": unfinished,
            "idle_teams": idle,
        })
    return out
