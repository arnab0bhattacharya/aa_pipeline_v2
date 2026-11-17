"""Stage `analyze`: computes empirical statistics from scored evaluation spaces.

Two outcomes per method (burrows, cosine, pca_full, pca_no_fw, pca_no_lex):
  attr_fail     closed-set attribution: nearest of the 10 centroids is not the target
  verif_reject  verification: distance to the target exceeds the target's LOO threshold
A text is flagged under an outcome when that outcome is True. Imitation rates are
evaluated against authentic baseline outcomes (LOO, LOWO, anchors).

Generated texts sharing an anchor (2 scenarios x 4 models) are clustered, so
imitation bootstrap CIs resample anchors as clusters. Authentic CIs resample chunks.

Outputs generated in out/results/:
  rq1_detection.csv        imitation vs authentic detection rates per method x outcome x author
  rq2_model_rates.csv      per-model detection rates
  rq2_model_contrasts.csv  pairwise model differences and TOST equivalence tests
  rq2_glm.csv              cluster-robust binomial GLM: detected ~ model + author
  rq3_dissociation.csv     Delta vs PCA 2x2 contingency counts and PCA-only rates
  rq4_burrows_cosine.csv   Burrows vs Cosine Delta metric comparison (McNemar exact test + Cohen's kappa)
  rq5_confusion.csv        target author x nearest author attribution confusion matrices
  controls_nearest.csv     unprompted controls: nearest author distribution
  controls_acceptance.csv  unprompted controls: acceptance rates under author thresholds
  controls_shift.csv       paired distance shift: control vs imitation of same scenario
  mfw_sensitivity.csv      attribution failure rates across most frequent words (MFW) sweep
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

from .config import Config
from .evaluate import full_space, load_corpus
from .space import DeltaModel, delta_loo

OUTCOMES = ("attr_fail", "verif_reject")


# ── bootstrap helpers ─────────────────────────────────────────────────────────

def _cluster_sums(x: np.ndarray, clusters: np.ndarray | None):
    if clusters is None:
        return x.astype(float), np.ones(len(x))
    s = pd.DataFrame({"x": x.astype(float), "c": clusters}).groupby("c")["x"].agg(["sum", "count"])
    return s["sum"].to_numpy(), s["count"].to_numpy(float)


def boot_draws(x: np.ndarray, clusters: np.ndarray | None, B: int, rng) -> np.ndarray:
    s, n = _cluster_sums(x, clusters)
    pick = rng.integers(0, len(s), size=(B, len(s)))
    return s[pick].sum(axis=1) / n[pick].sum(axis=1)


def ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2 * 100
    return float(np.percentile(draws, a)), float(np.percentile(draws, 100 - a))


# ── inputs ────────────────────────────────────────────────────────────────────

def _load(cfg: Config) -> dict[str, pd.DataFrame]:
    d = {k: pd.read_csv(cfg.out / f"{k}.csv", keep_default_na=False, na_values=[""])
         for k in ("loo", "lowo", "anchors", "generated", "thresholds")}
    thr = d["thresholds"].set_index(["method", "author"])["threshold"]
    loo = d["loo"]
    loo["threshold"] = [thr[(m, a)] for m, a in zip(loo["method"], loo["author"])]
    loo["verif_reject"] = loo["d_own"] > loo["threshold"]   # in-sample: ~5% by construction
    for k in ("loo", "lowo", "anchors", "generated"):
        for o in OUTCOMES:
            if o in d[k]:
                d[k][o] = d[k][o].map({True: True, False: False, "True": True, "False": False})
    d["imitations"] = d["generated"][d["generated"]["set"] == "imitation"].copy()
    d["controls"] = d["generated"][d["generated"]["set"] == "control"].copy()
    return d


# ── Imitation vs authentic detection rates (LOO, LOWO, anchors) ───────────────

def rq1(cfg: Config, d: dict, rng) -> pd.DataFrame:
    rows = []
    for (method, outcome), author in itertools.product(
            itertools.product(cfg.methods, OUTCOMES), ("ALL",) + cfg.authors):
        def sub(df):
            s = df[df["method"] == method]
            return s if author == "ALL" else s[s["author"] == author]
        imi = sub(d["imitations"])
        draws = {"imitation": boot_draws(imi[outcome].to_numpy(), imi["anchor_id"].to_numpy(), cfg.bootstrap, rng)}
        row = {"method": method, "outcome": outcome, "author": author, "n_imitation": len(imi),
               "imitation_rate": float(imi[outcome].mean())}
        row["imitation_lo"], row["imitation_hi"] = ci(draws["imitation"])
        for name in ("loo", "lowo", "anchors"):
            s = sub(d[name])
            draws[name] = boot_draws(s[outcome].to_numpy(), None, cfg.bootstrap, rng)
            row[f"n_{name}"], row[f"{name}_rate"] = len(s), float(s[outcome].mean())
            row[f"{name}_lo"], row[f"{name}_hi"] = ci(draws[name])
        for name in ("lowo", "anchors"):
            diff = draws["imitation"] - draws[name]
            row[f"diff_vs_{name}"] = row["imitation_rate"] - row[f"{name}_rate"]
            row[f"diff_vs_{name}_lo"], row[f"diff_vs_{name}_hi"] = ci(diff)
        if outcome == "attr_fail":
            row["closed_set_chance"] = (len(cfg.authors) - 1) / len(cfg.authors)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Per-model detection rates, pairwise TOST contrasts, and cluster GLM ───────

def rq2(cfg: Config, d: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates, contrasts, glm = [], [], []
    imi = d["imitations"]
    margin = cfg.equivalence_margin
    for method, outcome in itertools.product(cfg.methods, OUTCOMES):
        s = imi[imi["method"] == method]
        # per cluster x model sums: models are crossed with scenarios, so differences are paired
        piv = s.pivot_table(index="anchor_id", columns="model_slot", values=outcome, aggfunc=["sum", "count"])
        pick = rng.integers(0, len(piv), size=(cfg.bootstrap, len(piv)))
        boot = {m: piv[("sum", m)].to_numpy(float)[pick].sum(1) / piv[("count", m)].to_numpy(float)[pick].sum(1)
                for m in cfg.model_slots}
        for m in cfg.model_slots:
            r = float(s.loc[s["model_slot"] == m, outcome].mean())
            lo, hi = ci(boot[m])
            rates.append({"method": method, "outcome": outcome, "model_slot": m, "rate": r, "lo": lo, "hi": hi})
        for m1, m2 in itertools.combinations(cfg.model_slots, 2):
            diff = boot[m1] - boot[m2]
            est = float(s.loc[s["model_slot"] == m1, outcome].mean() - s.loc[s["model_slot"] == m2, outcome].mean())
            lo90, hi90 = ci(diff, 0.90)
            lo95, hi95 = ci(diff, 0.95)
            contrasts.append({"method": method, "outcome": outcome, "model_1": m1, "model_2": m2,
                              "diff": est, "ci95_lo": lo95, "ci95_hi": hi95, "ci90_lo": lo90, "ci90_hi": hi90,
                              "equivalent_tost": bool(lo90 > -margin and hi90 < margin)})
        glm.append(_glm(s, method, outcome))
    return pd.DataFrame(rates), pd.DataFrame(contrasts), pd.DataFrame(glm)


def _glm(s: pd.DataFrame, method: str, outcome: str) -> dict:
    df = s.assign(y=s[outcome].astype(int))
    row = {"method": method, "outcome": outcome, "n": len(df)}
    by_author = df.groupby("author")["y"].mean()
    degenerate = sorted(by_author[(by_author == 0) | (by_author == 1)].index)
    row["authors_at_0_or_100pct"] = ";".join(degenerate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smf.glm("y ~ C(model_slot) + C(author)", data=df, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(df["anchor_id"])[0]})
        tests = res.wald_test_terms(skip_single=False, scalar=True).table
    row["model_wald_p"] = float(tests.loc["C(model_slot)", "pvalue"])
    row["author_wald_p"] = float(tests.loc["C(author)", "pvalue"])
    row["warnings"] = " | ".join(sorted({str(w.message)[:120] for w in caught}))
    return row


# ── Delta vs PCA method contingency and dissociation ──────────────────────────

def _pair_rows(df: pd.DataFrame, delta: str, pca: str, kind: str):
    a = df[df["method"] == delta].set_index("doc_id")
    b = df[df["method"] == pca].set_index("doc_id").loc[a.index]
    if kind == "paper":   # submitted paper: Delta attribution vs PCA sphere-or-attribution
        dd, pp = a["attr_fail"], b["attr_fail"] | b["verif_reject"]
    else:
        dd, pp = a[kind], b[kind]
    return dd.to_numpy(bool), pp.to_numpy(bool), a


def rq3(cfg: Config, d: dict, rng) -> pd.DataFrame:
    rows = []
    for (delta, pca, kind), name in itertools.product(
            itertools.product(cfg.delta_methods, cfg.pca_levels, OUTCOMES + ("paper",)),
            ("imitations", "anchors", "lowo")):
        df = d[name]
        if name == "lowo":
            df = df.assign(doc_id=df["fold"] + "|" + df["doc_id"])
        dd, pp, a = _pair_rows(df, delta, pca, kind)
        pca_only = pp & ~dd
        clusters = a["anchor_id"].to_numpy() if name == "imitations" else None
        lo, hi = ci(boot_draws(pca_only, clusters, cfg.bootstrap, rng))
        rows.append({"delta": delta, "pca": pca, "outcome": kind, "set": name, "n": len(dd),
                     "both": int((dd & pp).sum()), "delta_only": int((dd & ~pp).sum()),
                     "pca_only": int(pca_only.sum()), "neither": int((~dd & ~pp).sum()),
                     "pca_only_rate": float(pca_only.mean()), "pca_only_lo": lo, "pca_only_hi": hi})
    return pd.DataFrame(rows)


# ── Comparison between Burrows' Delta and Cosine Delta (McNemar + Kappa) ──────

def rq4(cfg: Config, d: dict) -> pd.DataFrame:
    rows = []
    imi = d["imitations"]
    for outcome, author in itertools.product(OUTCOMES, ("ALL",) + cfg.authors):
        s = imi if author == "ALL" else imi[imi["author"] == author]
        b = s[s["method"] == "burrows"].set_index("doc_id")[outcome]
        c = s[s["method"] == "cosine"].set_index("doc_id")[outcome].loc[b.index]
        b, c = b.to_numpy(bool), c.to_numpy(bool)
        only_b, only_c = int((b & ~c).sum()), int((~b & c).sum())
        row = {"outcome": outcome, "author": author, "n": len(b), "burrows_rate": b.mean(),
               "cosine_rate": c.mean(), "both": int((b & c).sum()), "burrows_only": only_b,
               "cosine_only": only_c, "neither": int((~b & ~c).sum()),
               "agreement": float((b == c).mean()),
               "kappa": float(cohen_kappa_score(b, c)) if len(set(b) | set(c)) > 1 else float("nan")}
        if author == "ALL":
            row["mcnemar_exact_p"] = (binomtest(only_b, only_b + only_c, 0.5).pvalue
                                      if only_b + only_c else 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Nearest-author attribution confusion and unprompted control dynamics ──────

def rq5(d: dict) -> pd.DataFrame:
    out = []
    for name in ("imitations", "loo", "lowo", "anchors"):
        t = d[name].groupby(["method", "author", "nearest"]).size().rename("n").reset_index()
        out.append(t.rename(columns={"author": "target"}).assign(set=name))
    return pd.concat(out, ignore_index=True)


def controls(cfg: Config, d: dict, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con, imi = d["controls"], d["imitations"]
    nearest = con.groupby(["method", "nearest"]).size().rename("n").reset_index()
    nearest["share"] = nearest["n"] / nearest.groupby("method")["n"].transform("sum")

    acc = []
    for method, s in con.groupby("method"):
        for a in cfg.authors:
            acc.append({"method": method, "author": a, "accepted_by_author_threshold": float(s[f"accept_{a}"].mean())})

    # paired shift: control vs the imitation of the same scenario by the same model
    key = ["method", "scenario_id", "model_slot"]
    m = con.merge(imi, on=key, suffixes=("_con", "_imi"))
    m["d_target_control"] = [r[f"d_{r['scenario_author_con']}_con"] for _, r in m.iterrows()]
    m["d_target_imitation"] = m["d_own_imi"]
    m["shift"] = m["d_target_control"] - m["d_target_imitation"]
    shift = []
    for (method, author), s in itertools.chain(
            ((k, g) for k, g in m.groupby(["method", "scenario_author_con"])),
            (((k, "ALL"), g) for k, g in m.groupby("method"))):
        draws = boot_draws(s["shift"].to_numpy(), None, cfg.bootstrap, rng) if len(s) > 1 else np.array([np.nan])
        lo, hi = ci(draws)
        shift.append({"method": method, "author": author, "n_pairs": len(s), "mean_shift": float(s["shift"].mean()),
                      "lo": lo, "hi": hi, "share_imitation_closer": float((s["shift"] > 0).mean())})
    return nearest, pd.DataFrame(acc), pd.DataFrame(shift)


# ── Vocabulary sensitivity: Most Frequent Words (MFW) sweep ───────────────────

def mfw_sensitivity(cfg: Config) -> pd.DataFrame:
    docs, corpus = load_corpus(cfg)
    space, _ = full_space(cfg, docs, corpus)
    authors = list(cfg.authors)
    n_all = int(space.fw_mask.sum())
    fw_cols = np.where(space.fw_mask)[0]
    imi = docs[docs["set"] == "imitation"]
    X_imi = space.matrix(corpus.idx(imi["doc_id"]))
    target = imi["author"].to_numpy()
    rows = []
    for k in sorted({x for x in cfg.mfw_sweep if x < n_all} | {n_all}):
        cols = fw_cols[:k]
        model = DeltaModel(authors).fit(space.X_ref[:, cols], space.y)
        D = model.distances(X_imi[:, cols])
        L = delta_loo(space.X_ref[:, cols], space.y, authors)
        for method in cfg.delta_methods:
            fail = np.array(authors)[D[method].argmin(1)] != target
            loo_fail = np.array(authors)[L[method].argmin(1)] != space.y
            rows.append({"mfw": k, "method": method, "imitation_attr_fail": float(fail.mean()),
                         "loo_attr_fail": float(loo_fail.mean())})
            for m in cfg.model_slots:
                sel = imi["model_slot"].to_numpy() == m
                rows[-1][f"imitation_attr_fail_{m}"] = float(fail[sel].mean())
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    res = cfg.out / "results"
    res.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    d = _load(cfg)

    rq1(cfg, d, rng).to_csv(res / "rq1_detection.csv", index=False)
    rates, contrasts, glm = rq2(cfg, d, rng)
    rates.to_csv(res / "rq2_model_rates.csv", index=False)
    contrasts.to_csv(res / "rq2_model_contrasts.csv", index=False)
    glm.to_csv(res / "rq2_glm.csv", index=False)
    rq3(cfg, d, rng).to_csv(res / "rq3_dissociation.csv", index=False)
    rq4(cfg, d).to_csv(res / "rq4_burrows_cosine.csv", index=False)
    rq5(d).to_csv(res / "rq5_confusion.csv", index=False)
    nearest, acc, shift = controls(cfg, d, rng)
    nearest.to_csv(res / "controls_nearest.csv", index=False)
    acc.to_csv(res / "controls_acceptance.csv", index=False)
    shift.to_csv(res / "controls_shift.csv", index=False)
    mfw_sensitivity(cfg).to_csv(res / "mfw_sensitivity.csv", index=False)
    print(f"analyze: results written to {res}")
