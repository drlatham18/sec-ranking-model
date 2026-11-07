"""Export timestamped, reproducible ATS forecasts from the live model output.

Spread convention: home_spread is points added to the home team's final margin.
A negative spread means the home team is favored. Cover probabilities use the
model's normal residual distribution, NOT its straight-up win probability.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
SPREADS = ROOT / "data" / "contest_spreads.json"

def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def export():
    model = json.loads((OUT / "app_data.json").read_text())
    configured = json.loads(SPREADS.read_text()) if SPREADS.exists() else []
    by_match = {(s["home"], s["away"]): s for s in configured}
    now = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    rows = []
    for game in model["games"]:
        if game["played"]:
            continue
        spread = by_match.get((game["home"], game["away"]))
        margin = game.get("margin_home_current")
        sd = game.get("sd_current")
        # Do not silently use a preseason prediction as a live forecast.
        if margin is None or sd is None or sd <= 0:
            margin, sd = None, None
        line = spread["home_spread"] if spread else None
        p_home_cover = normal_cdf((margin + line) / sd) if (
            line is not None and margin is not None and sd is not None) else None
        rows.append({
            "game_id": game["id"], "kickoff": game["kickoff"],
            "home": game["home"], "away": game["away"],
            "neutral": game["neutral"], "forecast_basis": game.get("forecast_basis"),
            "projected_margin_home": margin, "residual_sd": sd,
            "home_win_probability": game.get("p_home_current"),
            "home_spread": line,
            "spread_source": spread.get("source") if spread else None,
            "home_cover_probability": round(p_home_cover, 4) if p_home_cover is not None else None,
            "away_cover_probability": round(1 - p_home_cover, 4) if p_home_cover is not None else None,
            "ats_pick": (game["home"] if p_home_cover > 0.5 else game["away"])
                        if p_home_cover is not None and p_home_cover != 0.5 else None,
            "spread_edge_points": round(margin + line, 2)
                                  if margin is not None and line is not None else None,
        })
    payload = {
        "schema_version": 1, "generated_at": now.isoformat(),
        "model_generated_at": model["generated_at"],
        "data_checked_at": model.get("data_source", {}).get("checked_at"),
        "season": model["season"], "model_conference": model["conference"],
        "model_basis": model.get("rating_basis"),
        "spread_convention": "home_spread is added to home margin",
        "spread_file": "data/contest_spreads.json",
        "note": "No cover probability is asserted when a spread or current-strength forecast is missing. Normal approximation; not separately ATS-calibrated.",
        "games": rows,
    }
    archive = OUT / "prediction_snapshots"
    archive.mkdir(exist_ok=True)
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    (archive / (stamp + ".json")).write_text(serialized)
    (OUT / "predictions_latest.json").write_text(serialized)
    print(f"[predictions] {len(rows)} future games; {sum(r['home_cover_probability'] is not None for r in rows)} ATS forecasts; snapshot {stamp}")
    return payload

if __name__ == "__main__":
    export()
