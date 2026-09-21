"""Inject output/app_data.json into ui/template.html -> ui/index.html."""
import os, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    tpl = (ROOT / "ui" / "template.html").read_text(encoding="utf-8")
    data = (ROOT / "output" / "app_data.json").read_text(encoding="utf-8")
    # guard against the JSON ending the inline <script> block early
    data = data.replace("</", r"<\/")
    out = ROOT / "ui" / "index.html"
    origin = os.environ.get("SITE_ORIGIN", "").rstrip("/")
    social_image = (origin + "/og.png") if origin else "og.png"
    bets_path = ROOT / 'output/best_bets.json'
    bets = bets_path.read_text(encoding='utf-8') if bets_path.exists() else json.dumps({
        'status': 'unavailable', 'rows': [], 'matched_games': 0, 'events_scanned': 0})
    rendered = tpl.replace("__DATA__", data).replace("__SOCIAL_IMAGE__", social_image).replace(
        '__BEST_BETS__', bets.replace('</', r'<\/'))
    out.write_text(rendered, encoding="utf-8")
    print("[ui] %s  (%d KB)" % (out, out.stat().st_size // 1024))
    return out


if __name__ == "__main__":
    build()
