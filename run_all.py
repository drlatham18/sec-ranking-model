"""End-to-end: build the panel, fit + validate, produce the ranking.

python run_all.py [start_season] [end_season]
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import build_dataset
import build_ui
import export_data
import fit
import fit_inseason
import rank


def main(start=2004, end=2026):
    print("=" * 70)
    print("STEP 1/6  build panel  (%d-%d)" % (start, end))
    print("=" * 70)
    build_dataset.build(start, end)

    print("\n" + "=" * 70)
    print("STEP 2/6  feature selection + untouched holdout validation")
    print("=" * 70)
    fit.run(2014)

    print("\n" + "=" * 70)
    print("STEP 3/6  ranking + explanations")
    print("=" * 70)
    rank.project(end, "SEC")

    print("\n" + "=" * 70)
    print("STEP 4/6  in-season blend: fit K, validate against the preseason baseline")
    print("=" * 70)
    fit_inseason.main()

    print("\n" + "=" * 70)
    print("STEP 5/6  schedule + season simulation")
    print("=" * 70)
    export_data.build(end, "SEC")

    print("\n" + "=" * 70)
    print("STEP 6/6  product UI")
    print("=" * 70)
    build_ui.build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", nargs="?", type=int, default=2004)
    parser.add_argument("end", nargs="?", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true", help="Bypass all API cache entries")
    args = parser.parse_args()
    if args.refresh:
        os.environ["CFBD_REFRESH"] = "1"
    main(args.start, args.end)
