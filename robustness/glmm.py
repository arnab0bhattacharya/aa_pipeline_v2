"""GLMM robustness check for RQ2 (Reviewer 1: "a mixed effects ... model would be more appropriate").

Model, per method x outcome, fitted on the 2,000 imitations:

    logit P(y=1) = b0 + model[m] + author[a] + u_anchor + v_scenario
    u_anchor ~ N(0, s_a^2)            250 anchors
    v_scenario ~ N(0, s_s^2)          500 scenarios, nested in anchor (2 per anchor, 4 models each)

Maximum likelihood with the random effects integrated out by nested Gauss-Hermite
quadrature (the same estimand as lme4::glmer with nAGQ > 1). Inference on the model and
author factors is by likelihood-ratio test (drop the term, refit). The latent-scale variance
partition (fixed author / fixed model / anchor / scenario / residual pi^2/3) follows
Nakagawa & Schielzeth (2013).

Run:  .venv\\Scripts\\python robustness\\glmm.py
Writes: out/results/rq2_glmm.csv, out/results/rq2_glmm_coefs.csv
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
from scipy.special import log_expit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "results"
METHODS = ["burrows", "cosine", "pca_full", "pca_no_fw", "pca_no_lex"]
OUTCOMES = ["attr_fail", "verif_reject"]
MODELS = ["M1", "M2", "M3", "M4"]
Q = 25  # quadrature nodes per random effect

gh_x, gh_w = np.polynomial.hermite.hermgauss(Q)
NODES = np.sqrt(2.0) * gh_x          # for N(0,1)
LOGW = np.log(gh_w / np.sqrt(np.pi))


class Data:
    def __init__(self, s: pd.DataFrame, terms: tuple[str, ...]):
        s = s.sort_values(["anchor_id", "scenario_id", "model_slot"]).reset_index(drop=True)
        self.y = s["y"].to_numpy(float).reshape(250, 2, 4)
        cols = [np.ones(len(s))]
        self.names = ["(Intercept)"]
        if "model" in terms:
            for m in MODELS[1:]:
                cols.append((s["model_slot"] == m).to_numpy(float)); self.names.append(f"model[{m}]")
        if "author" in terms:
            authors = sorted(s["author"].unique())
            for a in authors[1:]:
                cols.append((s["author"] == a).to_numpy(float)); self.names.append(f"author[{a}]")
        self.X = np.column_stack(cols)
        self.p = self.X.shape[1]
        self.frame = s


def negll_grad(theta: np.ndarray, D: Data, grad: bool = True):
    beta, log_sa, log_ss = theta[:D.p], theta[D.p], theta[D.p + 1]
    eta = (D.X @ beta).reshape(250, 2, 4)
    sa, ss = np.exp(log_sa), np.exp(log_ss)
    U = NODES[None, None, None, :, None]
    V = NODES[None, None, None, None, :]
    # eta + sa*u + ss*v  -> shape (anchor, scen, model, Qu, Qv)
    lin = eta[..., None, None] + sa * U + ss * V
    y = D.y[..., None, None]
    ll_obs = y * log_expit(lin) + (1 - y) * log_expit(-lin)
    a = ll_obs.sum(axis=2) + LOGW                                            # (A, S, Qu, Qv)
    ll_scen_given_u = np.logaddexp.reduce(a, axis=-1)                        # integrate v
    b = ll_scen_given_u.sum(axis=1) + LOGW                                   # (A, Qu)
    ll_anchor = np.logaddexp.reduce(b, axis=-1)                              # integrate u
    nll = -ll_anchor.sum()
    if not grad:
        return nll
    # posterior weights over quadrature nodes
    pu = np.exp(b - ll_anchor[:, None])                                      # (A, Qu)
    pv = np.exp(a - ll_scen_given_u[..., None])                              # (A, S, Qu, Qv)
    W = pu[:, None, :, None] * pv                                            # (A, S, Qu, Qv)
    r = y - 1.0 / (1.0 + np.exp(-lin))                                       # (A, S, M, Qu, Qv)
    Wr = W[:, :, None] * r
    g_beta = D.X.T @ Wr.sum(axis=(-2, -1)).reshape(-1)
    g_sa = (Wr * (sa * U)).sum()
    g_ss = (Wr * (ss * V)).sum()
    return nll, -np.r_[g_beta, g_sa, g_ss]


def negll(theta: np.ndarray, D: Data) -> float:
    return negll_grad(theta, D, grad=False)


def fit(s: pd.DataFrame, terms: tuple[str, ...]):
    D = Data(s, terms)
    glm = sm.GLM(D.frame["y"], D.X, family=sm.families.Binomial()).fit()
    start = np.r_[glm.params, np.log(0.5), np.log(0.5)]
    res = optimize.minimize(negll_grad, start, args=(D,), jac=True, method="L-BFGS-B",
                            bounds=[(-25, 25)] * D.p + [(-8, 3), (-8, 3)],
                            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7})
    return D, res


def hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n)); f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n); ei[i] = h; ej[j] = h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei) - f(x + ej) + f0) / h**2
    return H


def main():
    g = pd.read_csv(ROOT / "out" / "generated.csv")
    imi = g[g["set"] == "imitation"]
    summary, coefs = [], []
    for method, outcome in itertools.product(METHODS, OUTCOMES):
        s = imi[imi["method"] == method].assign(y=lambda d: d[outcome].astype(int))
        D, full = fit(s, ("model", "author"))
        _, no_model = fit(s, ("author",))
        _, no_author = fit(s, ("model",))
        lr_model = 2 * (no_model.fun - full.fun)
        lr_author = 2 * (no_author.fun - full.fun)
        beta = full.x[:D.p]; sa, ss = np.exp(full.x[D.p:])

        # Hessian by central differences of the analytic gradient
        n = len(full.x); H = np.zeros((n, n)); h = 1e-5
        for i in range(n):
            e = np.zeros(n); e[i] = h
            H[:, i] = (negll_grad(full.x + e, D)[1] - negll_grad(full.x - e, D)[1]) / (2 * h)
        H = (H + H.T) / 2
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov)[:D.p], 0, None))

        # latent-scale variance partition (Nakagawa & Schielzeth 2013)
        Xm = D.X[:, [i for i, n in enumerate(D.names) if n.startswith("model")]]
        Xa = D.X[:, [i for i, n in enumerate(D.names) if n.startswith("author")]]
        bm = beta[[i for i, n in enumerate(D.names) if n.startswith("model")]]
        ba = beta[[i for i, n in enumerate(D.names) if n.startswith("author")]]
        v_model, v_author = np.var(Xm @ bm), np.var(Xa @ ba)
        v_anchor, v_scen, v_res = sa**2, ss**2, np.pi**2 / 3
        tot = v_model + v_author + v_anchor + v_scen + v_res

        # cluster-robust GLM (the paper's current model) for side-by-side comparison
        cr = smf.glm("y ~ C(model_slot) + C(author)", data=s, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(s["anchor_id"])[0]})
        crt = cr.wald_test_terms(skip_single=False, scalar=True).table

        summary.append({
            "method": method, "outcome": outcome, "n": len(s), "converged": bool(full.success),
            "loglik": -full.fun, "sd_anchor": sa, "sd_scenario": ss,
            "LR_model_chi2": lr_model, "LR_model_df": 3, "LR_model_p": stats.chi2.sf(lr_model, 3),
            "LR_author_chi2": lr_author, "LR_author_df": 9, "LR_author_p": stats.chi2.sf(lr_author, 9),
            "share_author": v_author / tot, "share_model": v_model / tot,
            "share_anchor": v_anchor / tot, "share_scenario": v_scen / tot, "share_residual": v_res / tot,
            "author_to_model_ratio": v_author / v_model,
            "cr_model_wald_p": float(crt.loc["C(model_slot)", "pvalue"]),
            "cr_author_wald_p": float(crt.loc["C(author)", "pvalue"]),
        })
        for n, b, e in zip(D.names, beta, se):
            coefs.append({"method": method, "outcome": outcome, "term": n, "logit": b, "se": e,
                          "OR": np.exp(b), "OR_lo": np.exp(b - 1.96 * e), "OR_hi": np.exp(b + 1.96 * e),
                          "p": 2 * stats.norm.sf(abs(b / e)) if e > 0 else np.nan})
        print(f"{method:10s} {outcome:12s} sd_a={sa:.3f} sd_s={ss:.3f} "
              f"LR_model p={summary[-1]['LR_model_p']:.2e} LR_author p={summary[-1]['LR_author_p']:.2e} "
              f"author/model var={v_author / v_model:.1f}", flush=True)
    pd.DataFrame(summary).to_csv(OUT / "rq2_glmm.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "rq2_glmm_coefs.csv", index=False)


if __name__ == "__main__":
    main()
