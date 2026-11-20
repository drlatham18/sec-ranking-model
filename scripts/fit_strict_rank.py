"""Fit on earlier seasons, select on 2017-2022, freeze, test on 2023-2025.

Every feature snapshot excludes the week being predicted. FBS membership comes
from the historical panel, never its ratings. Targets are subsequent game
margins, never a poll. Nonnegative least squares learns component weights.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import strict_rank


def samples(games, membership, cap, shrink):
    features, outcomes, seasons = [], [], []
    for season in sorted(games.season.unique()):
        if not 2014 <= season <= 2025:
            continue
        teams = membership[season]
        sg = games[games.season == season]
        for week in sorted(sg.week.unique()):
            if week < 5:
                continue
            prior = sg[sg.week < week].to_dict('records')
            ev, _ = strict_rank.evidence(prior, teams, cap, shrink)
            for g in sg[sg.week == week].to_dict('records'):
                h, a = ev.get(g['home']), ev.get(g['away'])
                if h is None or a is None:
                    continue
                features.append([h['production']-a['production'], h['sor_score']-a['sor_score'],
                                 0. if g['neutral'] else 1.])
                outcomes.append(g['home_points']-g['away_points'])
                seasons.append(season)
    return np.array(features), np.array(outcomes), np.array(seasons)


def metrics(y, p):
    return dict(games=len(y), rmse=float(np.sqrt(np.mean((y-p)**2))),
                mae=float(np.mean(abs(y-p))), accuracy=float(np.mean((y>0)==(p>0))))


def main():
    games = pd.read_csv(ROOT/'data/processed/games.csv').dropna(subset=['home_points','away_points'])
    games = games.rename(columns={'home_team':'home','away_team':'away'})
    panel = pd.read_csv(ROOT/'data/processed/panel.csv')
    members = {int(s): set(g.team) for s, g in panel.groupby('season')}
    candidates = []
    arrays = {}
    for cap in (21, 28, 35):
        for shrink in (1., 3.):
            X,y,S = samples(games, members, cap, shrink)
            arrays[(cap,shrink)] = (X,y,S)
            actual, predicted = [], []
            for year in range(2017, 2023):
                b = nnls(X[S<year], y[S<year])[0]
                mask = S == year
                actual.extend(y[mask]); predicted.extend(X[mask]@b)
            result = dict(margin_cap=cap, shrinkage=shrink,
                          development=metrics(np.array(actual), np.array(predicted)))
            candidates.append(result)
            print(result, flush=True)
    chosen = min(candidates, key=lambda r: r['development']['rmse'])
    X,y,S = arrays[(chosen['margin_cap'],chosen['shrinkage'])]
    train, test = S<=2022, S>=2023
    b = nnls(X[train],y[train])[0]
    # Frozen production-only baseline is fitted on the same development data.
    baseline = nnls(X[train][:,[0,2]], y[train])[0]
    validation = dict(composite=metrics(y[test],X[test]@b),
                      production_only=metrics(y[test],X[test][:,[0,2]]@baseline))
    if min(b[:2]) <= 0:
        raise RuntimeError('Data did not support both requested components')
    out = dict(version='strict-results-v1', margin_cap=chosen['margin_cap'],
               shrinkage=chosen['shrinkage'],
               coefficients=dict(zip(['production','sor_score','home_field'],map(float,b))),
               trained_through=2022, development_seasons=list(range(2017,2023)),
               holdout_seasons=[2023,2024,2025], holdout=validation, candidates=candidates,
               minimum_validation_week=5,
               method='Results-only scoring production plus strength of record; weights learned from future game margins.',
               limitations=['Scoring production, not play-by-play efficiency or garbage-time-adjusted EPA.',
                            'Unrated opponents share a fitted pool; individual FCS strength is not resolved.',
                            'Early-season rankings have limited evidence; validation covers week 5 onward.',
                            'A resume ranking is not a calibrated betting probability.'])
    path=ROOT/'output/strict_ranking_calibration.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__ == '__main__': main()
