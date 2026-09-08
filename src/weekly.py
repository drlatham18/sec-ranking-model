"""Comparable weekly season outlooks using the same frozen preseason model."""
import hashlib
import numpy as np


def simulate(games, teams, n_sims=20000, through_week=None):
    names = sorted(teams)
    index = {team: i for i, team in enumerate(names)}
    wins = np.zeros((n_sims, len(names)), dtype=int)
    conf_wins = np.zeros_like(wins)
    records = {t: dict(wins=0, losses=0, conf_wins=0, conf_losses=0,
                       scheduled_games=0, scheduled_conf_games=0, unrated_opponents=0,
                       proj_wins=0., proj_conf_wins=0.) for t in names}
    for game in games:
        final = game["played"] and (through_week is None or game["week"] <= through_week)
        p = float(game["home_points"] > game["away_points"]) if final else game["p_home"]
        # Stable draws per matchup prevent unrelated refreshes from adding noise.
        identity = f'{game["home"]}|{game["away"]}|{game["week"]}'
        seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
        draw = np.random.default_rng(seed).random(n_sims) < p
        for home, team in ((True, game["home"]), (False, game["away"])):
            if team not in index:
                continue
            won = draw if home else ~draw
            expected = p if home else 1 - p
            stats = records[team]
            wins[:, index[team]] += won
            stats["scheduled_games"] += 1
            stats["proj_wins"] += expected
            stats["unrated_opponents"] += int(game.get("unrated", False))
            if final:
                stats["wins"] += int(expected)
                stats["losses"] += int(1 - expected)
            if game["conf_game"]:
                conf_wins[:, index[team]] += won
                stats["scheduled_conf_games"] += 1
                stats["proj_conf_wins"] += expected
                if final:
                    stats["conf_wins"] += int(expected)
                    stats["conf_losses"] += int(1 - expected)
    leaders = conf_wins == conf_wins.max(axis=1, keepdims=True)
    shares = leaders / leaders.sum(axis=1, keepdims=True)
    for team, stats in records.items():
        i = index[team]
        stats.update({"team": team, "proj_wins": round(stats["proj_wins"], 4),
                      "proj_conf_wins": round(stats["proj_conf_wins"], 4),
                      "wins_p10": int(np.percentile(wins[:, i], 10)),
                      "wins_p90": int(np.percentile(wins[:, i], 90)),
                      "win_distribution": [round(float(x), 4) for x in
                          np.bincount(wins[:, i], minlength=14)[:14] / n_sims],
                      "p_conf_leader": round(float(shares[:, i].mean()), 4)})
    # Equal projected win totals share a rank; alphabetical order only breaks display ties.
    order = sorted(names, key=lambda t: (-records[t]["proj_wins"], t))
    for i, team in enumerate(order):
        prev = records[order[i - 1]] if i else None
        records[team]["outlook_rank"] = (prev["outlook_rank"] if prev and
            prev["proj_wins"] == records[team]["proj_wins"] else i + 1)
    return list(records.values())


def history(games, teams, n_sims=20000):
    """Reconstruct each closed SEC week; never label a partial week complete."""
    snapshots = [{"week": -1, "label": "Preseason", "complete": True,
                  "teams": simulate(games, teams, n_sims, through_week=-1)}]
    weeks = sorted({g["week"] for g in games})
    for week in weeks:
        through = [g for g in games if g["week"] <= week]
        if not all(g["played"] for g in through):
            break
        snapshots.append({"week": week, "label": f"Week {week}", "complete": True,
                          "teams": simulate(games, teams, n_sims, through_week=week)})
    played_weeks = [g["week"] for g in games if g["played"]]
    if played_weeks and max(played_weeks) > snapshots[-1]["week"]:
        snapshots.append({"week": max(played_weeks), "label": "Current (partial week)",
                          "complete": False, "teams": simulate(games, teams, n_sims)})
    return snapshots
