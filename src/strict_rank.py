"""Results-only national ranking; no polls, recruiting, or preseason strength.

Production is capped scoring margin adjusted simultaneously for opponents and
venue. Record strength is the exact probability of a reference top-25 team
matching or exceeding the observed wins on the played schedule. Opponent win
strength is estimated by regularized Bradley-Terry, independently of margins.
All tuning and composite coefficients are frozen by fit_strict_rank.py.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.special import expit
from scipy.stats import norm

FCS = '__unrated_opponents__'
CALIBRATION = Path(__file__).resolve().parents[1] / 'output/strict_ranking_calibration.json'


def win_tail(probabilities, wins):
    """Poisson-binomial P(W >= observed wins), retaining full precision."""
    distribution = np.array([1.0])
    for p in probabilities:
        distribution = np.convolve(distribution, [1-p, p])
    return float(distribution[int(wins):].sum())


def evidence(games, teams, cap, shrink):
    teams = sorted(set(teams))
    names = teams + [FCS]
    index = {t: i for i, t in enumerate(names)}
    n = len(names)
    rows = []
    seen = set()
    for g in games:
        h, a = g['home'], g['away']
        hp, ap = g.get('home_points'), g.get('away_points')
        if g.get('completed', True) is False or g.get('played', True) is False:
            continue
        if hp is None or ap is None or not np.isfinite([hp, ap]).all():
            continue
        if h not in index and a not in index:
            continue
        key = g.get('id') or (h, a, g.get('week'))
        if key in seen:
            raise ValueError('Duplicate completed game in national ranking')
        seen.add(key)
        rows.append((index.get(h, n-1), index.get(a, n-1), float(hp), float(ap),
                     0.0 if g.get('neutral') else 1.0))
    if not rows:
        return {}, {'games': 0}
    data = np.array(rows)
    hi, ai = data[:, 0].astype(int), data[:, 1].astype(int)
    margin, hf = data[:, 2]-data[:, 3], data[:, 4]
    m = len(rows)
    X = csr_matrix((np.r_[np.ones(m), -np.ones(m), hf],
                    (np.r_[np.arange(m), np.arange(m), np.arange(m)],
                     np.r_[hi, ai, np.full(m, n)])), shape=(m, n+1))
    # Team shrinkage handles sparse/disconnected schedules. Venue is estimated.
    penalty = np.r_[np.full(n, shrink), shrink]
    prod = spsolve(X.T @ X + diags(penalty), X.T @ np.clip(margin, -cap, cap))
    prod[:n] -= np.mean(prod[:len(teams)])
    y = np.where(margin > 0, 1., np.where(margin < 0, 0., .5))
    def objective(b):
        z = X @ b
        loss = np.logaddexp(0, z).sum() - y @ z + .5*np.sum(penalty*b*b)
        grad = np.asarray(X.T @ (expit(z)-y)) + penalty*b
        return loss, grad
    fit = minimize(objective, np.zeros(n+1), jac=True, method='L-BFGS-B',
                   options={'ftol': 1e-11, 'gtol': 1e-7, 'maxiter': 400})
    if not fit.success:
        raise ValueError('Record strength fit failed: '+fit.message)
    record = fit.x
    counts = np.bincount(np.r_[hi, ai], minlength=n)
    eligible = [i for i in range(len(teams)) if counts[i]]
    benchmark = float(np.mean(sorted(record[eligible], reverse=True)[:25]))
    result = {}
    for t in teams:
        i = index[t]
        positions = np.flatnonzero((hi == i) | (ai == i))
        if not len(positions):
            continue  # no games means no earned rank
        home = hi[positions] == i
        opp = np.where(home, ai[positions], hi[positions])
        venue = np.where(home, hf[positions], -hf[positions])
        own_margin = np.where(home, margin[positions], -margin[positions])
        wins, losses, ties = int(sum(own_margin > 0)), int(sum(own_margin < 0)), int(sum(own_margin == 0))
        p = expit(benchmark - record[opp] + record[n]*venue)
        # Modern CFB has no ties; historical ties count as half a win in the
        # record fit, but use the lower integer threshold for a conservative SOR.
        tail = win_tail(p, wins)
        score = float(norm.isf(np.clip(tail, 1e-12, 1-1e-12)))
        score = max(score, -4.)  # numerical lower floor; all-win tails remain distinct
        result[t] = dict(production=float(prod[i]), sor_score=score,
            sor_probability=tail, schedule_strength=float(np.mean(prod[opp]-prod[n]*venue)),
            expected_reference_wins=float(p.sum()), wins=wins, losses=losses, ties=ties,
            record=f'{wins}-{losses}' + (f'-{ties}' if ties else ''), games_rated=len(positions),
            fcs_games=int(sum(opp == n-1)),
            points_for=float(np.mean(np.where(home, data[positions, 2], data[positions, 3]))),
            points_against=float(np.mean(np.where(home, data[positions, 3], data[positions, 2]))))
    return result, dict(games=m, production_home_field=float(prod[n]),
                       record_home_field=float(record[n]), reference_log_strength=benchmark)


def load_calibration():
    data = json.loads(CALIBRATION.read_text())
    if data['coefficients']['production'] <= 0 or data['coefficients']['sor_score'] <= 0:
        raise ValueError('Strict ranking requires positive production and record evidence')
    return data


def rank(games, teams, calibration=None):
    c = calibration or load_calibration()
    rows, meta = evidence(games, teams, c['margin_cap'], c['shrinkage'])
    for t, r in rows.items():
        r['team'] = t
        r['rating'] = (c['coefficients']['production']*r['production']
                       + c['coefficients']['sor_score']*r['sor_score'])
    ordered = sorted(rows.values(), key=lambda r: (-r['rating'], r['team']))
    for key, column in [('production', 'production_rank'), ('sor_score', 'sor_rank'),
                        ('schedule_strength', 'sos_rank')]:
        ranks = {r['team']: i+1 for i, r in enumerate(sorted(rows.values(), key=lambda r: (-r[key], r['team'])))}
        for r in ordered:
            r[column] = ranks[r['team']]
    for i, r in enumerate(ordered, 1):
        r.update(rank=i, from_results=1.0, previous_rank=None,
                 change_since_last_week=None, change_since_preseason=None, preseason_rank=None)
    return ordered, meta
