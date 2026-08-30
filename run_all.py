"""End-to-end: build the panel, fit + validate, produce the ranking.

python run_all.py [start_season] [end_season]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import build_dataset
import build_ui
import export_data
import fit
import rank


def main(start=2004, end=2026):
    print("=" * 70)
    print("STEP 1/5  build panel  (%d-%d)" % (start, end))
    print("=" * 70)
    build_dataset.build(start, end)

    print("\n" + "=" * 70)
    print("STEP 2/5  feature selection + untouched holdout validation")
    print("=" * 70)
    fit.run(2014)

    print("\n" + "=" * 70)
    print("STEP 3/5  ranking + explanations")
    print("=" * 70)
    rank.project(end, "SEC")

    print("\n" + "=" * 70)
    print("STEP 4/5  schedule + season simulation")
    print("=" * 70)
    export_data.build(end, "SEC")

    print("\n" + "=" * 70)
    print("STEP 5/5  product UI")
    print("=" * 70)
    build_ui.build()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2004, int(a[1]) if len(a) > 1 else 2026)
