"""The model's own national ranking.

A poll is votes. This is the rating, ordered. The only editorial decision is
where to cut the list off.

Each row carries what the number is actually made of -- how much of the rating
still comes from the preseason projection versus games played, and how the team
has moved since last week and since the preseason -- so a rank can be argued
with rather than just read.
"""


def records(games, teams):
    """Win-loss record per team from completed games.

    Counts every completed game, including ones against unrated opponents:
    a win is a win for the record even when it cannot inform a rating.
    """
    table = {t: {"wins": 0, "losses": 0, "ties": 0} for t in teams}
    for g in games:
        hp, ap = g.get("home_points"), g.get("away_points")
        if hp is None or ap is None:
            continue
        for team, own, opp in ((g["home"], hp, ap), (g["away"], ap, hp)):
            if team not in table:
                continue
            key = "wins" if own > opp else "losses" if own < opp else "ties"
            table[team][key] += 1
    for team, row in table.items():
        row["record"] = ("%d-%d" % (row["wins"], row["losses"])
                         + ("-%d" % row["ties"] if row["ties"] else ""))
    return table


def _ranks(ratings):
    return {t: i for i, (t, _) in
            enumerate(sorted(ratings.items(), key=lambda kv: -kv[1]), start=1)}


def table(current, preseason, previous=None, record_table=None,
          detail=None, limit=None):
    """Ordered ranking rows.

    current:   {team: rating now}
    preseason: {team: frozen preseason rating}
    previous:  {team: rating as of last week}, for movement
    detail:    {team: {"n_games", "weight"}} from the in-season blend
    """
    now = _ranks(current)
    pre = _ranks(preseason)
    prev = _ranks(previous) if previous else {}
    rows = []
    for team, rank in sorted(now.items(), key=lambda kv: kv[1]):
        if limit and rank > limit:
            break
        info = (detail or {}).get(team, {})
        rec = (record_table or {}).get(team, {})
        rows.append({
            "rank": rank,
            "team": team,
            "rating": round(current[team], 2),
            "record": rec.get("record"),
            "wins": rec.get("wins"),
            "losses": rec.get("losses"),
            "preseason_rank": pre.get(team),
            # positive = climbed since the preseason projection
            "change_since_preseason": (pre[team] - rank) if team in pre else None,
            "previous_rank": prev.get(team),
            "change_since_last_week": (prev[team] - rank) if team in prev else None,
            "preseason_rating": round(preseason[team], 2) if team in preseason else None,
            "games_rated": info.get("n_games"),
            # share of the rating driven by this season rather than the forecast
            "from_results": info.get("weight"),
        })
    return rows
