"""Statistical helpers used by H1-H3 (all unit-tested on synthetic data).

Conventions
* Matrices are square (n, n), symmetric; analyses use the strict upper triangle.
* Permutation tests permute **concepts** (rows and columns jointly), never
  individual pairs, so the dependence among pairs that share a concept is kept
  (Mantel 1967; DECISIONS.md D-19). p = (k + 1) / (N + 1).
* Bootstrap CIs resample **concepts** with replacement and drop pairs formed by
  two copies of the same concept (D-19), percentile intervals unless stated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as ss
from scipy.spatial import procrustes


# ------------------------------------------------------------------ basics --
def upper(m: np.ndarray, k: int = 1) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], k)]


def rank_square(m: np.ndarray) -> np.ndarray:
    """Square matrix holding the average ranks of the upper-triangle entries."""
    n = m.shape[0]
    iu = np.triu_indices(n, 1)
    r = np.zeros_like(m, dtype=float)
    r[iu] = ss.rankdata(m[iu])
    return r + r.T


def _z(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    sd = np.sqrt((v * v).sum())
    return v / sd if sd > 0 else v


def spearman_upper(a: np.ndarray, b: np.ndarray) -> float:
    return float(ss.spearmanr(upper(a), upper(b)).statistic)


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN entries ignored and kept NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    if q.size == 0:
        return out
    order = np.argsort(q)
    m = q.size
    adj = q[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


# ------------------------------------------------------------------ Mantel --
@dataclass
class MantelResult:
    r: float
    p_greater: float
    p_two_sided: float
    n_perm: int
    null_mean: float
    null_sd: float


def mantel(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator,
           method: str = "spearman") -> MantelResult:
    """Mantel test of the correlation between the upper triangles of x and y.
    Spearman = Pearson on ranks; y's concepts are permuted jointly."""
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    if method == "spearman":
        xs, ys = rank_square(x), rank_square(y)
    elif method == "pearson":
        xs, ys = np.asarray(x, float), np.asarray(y, float)
    else:
        raise ValueError(method)
    zx = _z(xs[iu])
    r_obs = float(zx @ _z(ys[iu]))
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = zx @ _z(ys[np.ix_(p, p)][iu])
    return MantelResult(
        r=r_obs,
        p_greater=float((np.sum(null >= r_obs) + 1) / (n_perm + 1)),
        p_two_sided=float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)),
        n_perm=n_perm, null_mean=float(null.mean()), null_sd=float(null.std()),
    )


def residualise_square(m: np.ndarray, controls: Sequence[np.ndarray]) -> np.ndarray:
    """Residuals of the upper-triangle ranks of m on the ranks of the controls
    (OLS with intercept), returned as a symmetric square matrix."""
    n = m.shape[0]
    iu = np.triu_indices(n, 1)
    y = rank_square(m)[iu]
    X = np.column_stack([np.ones_like(y)] + [rank_square(c)[iu] for c in controls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = np.zeros((n, n))
    out[iu] = y - X @ beta
    return out + out.T


def partial_mantel(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray], n_perm: int,
                   rng: np.random.Generator) -> MantelResult:
    """Partial Mantel on ranks (Smouse, Long & Sokal 1986): correlate the residuals of
    x and y after regressing each on the controls; permute y's residual matrix."""
    return mantel(residualise_square(x, controls), residualise_square(y, controls), n_perm, rng,
                  method="pearson")


# --------------------------------------------------------------- bootstrap --
def bootstrap_pairs(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One concept-level bootstrap draw. Returns (sampled concept indices, row
    and column index arrays of the retained pairs) — pairs of two copies of the
    same original concept are dropped."""
    idx = rng.integers(0, n, n)
    a, b = np.triu_indices(n, 1)
    keep = idx[a] != idx[b]
    return idx, idx[a][keep], idx[b][keep]


def bootstrap_ci(stat: Callable[[np.ndarray, np.ndarray], float], n: int, n_boot: int,
                 rng: np.random.Generator, level: float = 0.95) -> tuple[float, float, np.ndarray]:
    """``stat(rows, cols)`` computes the statistic on the given pair indices."""
    draws = np.empty(n_boot)
    for k in range(n_boot):
        _, r, c = bootstrap_pairs(n, rng)
        draws[k] = stat(r, c)
    lo, hi = np.nanpercentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi), draws


def rsa_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    lo, hi, _ = bootstrap_ci(lambda r, c: ss.spearmanr(x[r, c], y[r, c]).statistic, x.shape[0], n_boot, rng)
    return lo, hi


# ------------------------------------------------------------- regression --
def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with intercept prepended. Returns (beta incl. intercept, residuals, design)."""
    D = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return beta, y - D @ beta, D


def _t_values(y: np.ndarray, D: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = y - D @ beta
    s2 = res @ res / (len(y) - D.shape[1])
    cov = s2 * np.linalg.pinv(D.T @ D)
    return beta / np.sqrt(np.diag(cov))


def mrqap_dsp(y_sq: np.ndarray, X_sq: Sequence[np.ndarray], n_perm: int, rng: np.random.Generator
              ) -> np.ndarray:
    """MRQAP with double semi-partialling (Dekker, Krackhardt & Snijders 2007).
    For each predictor k: residualise it on the other predictors, permute the
    residual matrix's concepts jointly, refit, and compare |t|. Returns two-sided
    p-values, one per predictor."""
    n = y_sq.shape[0]
    iu = np.triu_indices(n, 1)
    y = y_sq[iu]
    Xs = [x[iu] for x in X_sq]
    X = np.column_stack(Xs)
    t_obs = _t_values(y, np.column_stack([np.ones(len(y)), X]))[1:]
    pvals = np.empty(len(Xs))
    N = len(y)
    for k in range(len(Xs)):
        others = np.column_stack([np.ones(N)] + [Xs[j] for j in range(len(Xs)) if j != k])
        proj = np.linalg.pinv(others)                       # (p-1, N), fixed across permutations
        e = np.zeros((n, n))
        e[iu] = Xs[k] - others @ (proj @ Xs[k])
        e = e + e.T
        ry = y - others @ (proj @ y)                        # y residualised on the other predictors
        dof = N - others.shape[1] - 1
        exceed = 0
        for _ in range(n_perm):
            p = rng.permutation(n)
            ep = e[np.ix_(p, p)][iu]
            re = ep - others @ (proj @ ep)                  # Frisch-Waugh-Lovell: t of ep in the full model
            ss_e = re @ re
            beta = (re @ ry) / ss_e
            s2 = (ry @ ry - beta * beta * ss_e) / dof
            exceed += abs(beta / np.sqrt(s2 / ss_e)) >= abs(t_obs[k])
        pvals[k] = (exceed + 1) / (n_perm + 1)
    return pvals


def dyadic_cluster_se(D: np.ndarray, resid: np.ndarray, a: np.ndarray, b: np.ndarray, n_nodes: int) -> np.ndarray:
    """Dyadic cluster-robust standard errors (Aronow, Samii & Assenova 2015):
    residual covariance allowed between any two pairs that share a concept."""
    xe = D * resid[:, None]
    S = np.zeros((n_nodes, D.shape[1]))
    np.add.at(S, a, xe)
    np.add.at(S, b, xe)
    meat = S.T @ S - xe.T @ xe          # each pair would otherwise be counted twice with itself
    bread = np.linalg.pinv(D.T @ D)
    return np.sqrt(np.clip(np.diag(bread @ meat @ bread), 0, None))


# ------------------------------------------------------------- Procrustes --
def protest(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Procrustes m² (scipy disparity, after centring, scaling and rotation) and the
    PROTEST permutation p-value (Jackson 1995): rows of b are permuted."""
    _, _, m2 = procrustes(a, b)
    null = np.array([procrustes(a, b[rng.permutation(len(b))])[2] for _ in range(n_perm)])
    return float(m2), float((np.sum(null <= m2) + 1) / (n_perm + 1))


# ------------------------------------------------------------- effect size --
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) - P(X < Y), computed exactly from ranks in O(n log n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    r = ss.rankdata(np.concatenate([x, y]))
    u = r[:nx].sum() - nx * (nx + 1) / 2          # Mann-Whitney U for x (ties count 1/2)
    return float(2 * u / (nx * ny) - 1)


def cliffs_delta_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator,
                    level: float = 0.95) -> tuple[float, float]:
    """BCa bootstrap CI (scipy) for Cliff's delta, resampling each group independently."""
    res = ss.bootstrap((np.asarray(x, float), np.asarray(y, float)), cliffs_delta, n_resamples=n_boot,
                       confidence_level=level, method="BCa", random_state=rng, vectorized=False)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)
