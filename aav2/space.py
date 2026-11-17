"""Vocabulary selection, feature matrices, and the Delta and PCA models.

Everything that depends on the reference set is fitted here, so the same code
serves the full reference set, every LOO refit and every LOWO fold:

  vocabulary   MFW = words in >= 50% of reference chunks (non-zero variance), capped
               at max_mfw; word/POS n-grams = the most frequent n-grams passing the
               same doc-frequency rule, capped at their configured counts
  Delta        z-scores from reference mean/std (population), author centroid =
               mean z-vector; Burrows = mean |z - c|; Cosine = cosine distance
  PCA          features with > 5% NaN in the reference dropped, remaining NaNs
               imputed with the reference mean, zero-variance columns dropped,
               StandardScaler, PCA to 95% variance; centroid = mean PC vector;
               distance = Manhattan / n_components

Feature groups (for the MFW ablation): fw (MFW frequencies), wngram (word bi/trigrams),
pos (POS bi/trigrams), other (the 29 fixed features).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import FIXED_FEATURES

KINDS = {  # kind: (sequence, n, feature prefix, group, config cap attribute)
    "fw": ("tok", 1, "fw_", "fw", "max_mfw"),
    "wbi": ("tok", 2, "wbi_", "wngram", "num_word_bigrams"),
    "wtri": ("tok", 3, "wtri_", "wngram", "num_word_trigrams"),
    "posbi": ("pos", 2, "posbi_", "pos", "num_pos_bigrams"),
    "postri": ("pos", 3, "postri_", "pos", "num_pos_trigrams"),
}


class Corpus:
    """All documents as integer-coded token and POS sequences plus fixed features."""

    def __init__(self, docs: pd.DataFrame, sequences: dict, fixed: pd.DataFrame):
        self.docs = docs.reset_index(drop=True)
        self.ids = list(self.docs["doc_id"])
        self.index = {d: i for i, d in enumerate(self.ids)}
        words = sorted({w for d in self.ids for w in sequences[d][0]})
        tags = sorted({p for d in self.ids for p in sequences[d][1]})
        self.words, self.tags = words, tags
        w_id = {w: i for i, w in enumerate(words)}
        p_id = {p: i for i, p in enumerate(tags)}
        self.seq = {
            "tok": [np.array([w_id[w] for w in sequences[d][0]], dtype=np.int64) for d in self.ids],
            "pos": [np.array([p_id[p] for p in sequences[d][1]], dtype=np.int64) for d in self.ids],
        }
        self.base = {"tok": len(words), "pos": len(tags)}
        self.fixed = fixed.set_index("doc_id").loc[self.ids, FIXED_FEATURES].to_numpy(dtype=float)

    def idx(self, doc_ids) -> np.ndarray:
        return np.array([self.index[d] for d in doc_ids], dtype=int)

    def codes(self, i: int, kind: str) -> np.ndarray:
        seq_name, n, *_ = KINDS[kind]
        s, base = self.seq[seq_name][i], self.base[seq_name]
        if len(s) < n:
            return np.empty(0, dtype=np.int64)
        code = s[: len(s) - n + 1].copy()
        for k in range(1, n):
            code = code * base + s[k: len(s) - n + 1 + k]
        return code

    def decode(self, kind: str, code: int) -> str:
        seq_name, n, *_ = KINDS[kind]
        base = self.base[seq_name]
        names = self.words if seq_name == "tok" else self.tags
        parts = []
        for _ in range(n):
            code, r = divmod(int(code), base)
            parts.append(names[r])
        return "_".join(reversed(parts))

    def rel_freq(self, idx: np.ndarray, kind: str, vocab_codes: np.ndarray) -> np.ndarray:
        """Relative frequency of each vocabulary item in each document (count / n-gram total)."""
        out = np.zeros((len(idx), len(vocab_codes)))
        order = np.argsort(vocab_codes)
        sorted_codes = vocab_codes[order]
        for r, i in enumerate(idx):
            c = self.codes(i, kind)
            if len(c) == 0:
                continue
            u, cnt = np.unique(c, return_counts=True)
            pos = np.searchsorted(sorted_codes, u)
            pos[pos == len(sorted_codes)] = 0
            hit = sorted_codes[pos] == u if len(sorted_codes) else np.zeros(len(u), bool)
            out[r, order[pos[hit]]] = cnt[hit] / len(c)
        return out


@dataclass
class Vocab:
    codes: dict[str, np.ndarray]
    names: dict[str, list[str]]

    def columns(self) -> tuple[list[str], list[str]]:
        cols, groups = list(FIXED_FEATURES), ["other"] * len(FIXED_FEATURES)
        for kind, (_, _, prefix, group, _) in KINDS.items():
            cols += [prefix + n for n in self.names[kind]]
            groups += [group] * len(self.names[kind])
        return cols, groups


def select_vocab(corpus: Corpus, ref_idx: np.ndarray, cfg: Config) -> Vocab:
    min_df = math.ceil(cfg.min_doc_freq_ratio * len(ref_idx))
    codes, names = {}, {}
    for kind, (_, _, _, _, cap_attr) in KINDS.items():
        cap = getattr(cfg, cap_attr)
        per_doc = [corpus.codes(i, kind) for i in ref_idx]
        u_all, total = np.unique(np.concatenate(per_doc), return_counts=True)
        u_df, df = np.unique(np.concatenate([np.unique(c) for c in per_doc]), return_counts=True)
        df_of = dict(zip(u_df.tolist(), df.tolist()))
        # most frequent first; ties broken by code for determinism
        order = np.lexsort((u_all, -total))
        cand = np.array([c for c in u_all[order] if df_of[c] >= min_df], dtype=np.int64)
        if len(cand):
            freqs = corpus.rel_freq(ref_idx, kind, cand)
            cand = cand[freqs.std(axis=0) > 0]
        codes[kind] = cand[:cap]
        names[kind] = [corpus.decode(kind, c) for c in codes[kind]]
    return Vocab(codes, names)


def feature_matrix(corpus: Corpus, idx: np.ndarray, vocab: Vocab) -> np.ndarray:
    blocks = [corpus.fixed[idx]]
    for kind in KINDS:
        blocks.append(corpus.rel_freq(idx, kind, vocab.codes[kind]))
    return np.hstack(blocks)


# ── Delta ─────────────────────────────────────────────────────────────────────

def _delta_distances(Z: np.ndarray, C: np.ndarray) -> dict[str, np.ndarray]:
    return {"burrows": cdist(Z, C, metric="cityblock") / Z.shape[1],
            "cosine": cdist(Z, C, metric="cosine")}


@dataclass
class DeltaModel:
    authors: list[str]
    mu: np.ndarray = field(init=False)
    sd: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeltaModel":
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        if (self.sd == 0).any():
            raise ValueError("Zero-variance MFW column in Delta reference")
        Z = (X - self.mu) / self.sd
        self.C = np.vstack([Z[y == a].mean(axis=0) for a in self.authors])
        return self

    def distances(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return _delta_distances((X - self.mu) / self.sd, self.C)


def delta_loo(X: np.ndarray, y: np.ndarray, authors: list[str]) -> dict[str, np.ndarray]:
    """Exact leave-one-out Delta: z-statistics and centroids recomputed without chunk i."""
    n = len(X)
    S, Q = X.sum(axis=0), (X ** 2).sum(axis=0)
    mu = (S - X) / (n - 1)
    sd = np.sqrt((Q - X ** 2) / (n - 1) - mu ** 2)
    sums = np.vstack([X[y == a].sum(axis=0) for a in authors])          # A x k
    cnt = np.array([(y == a).sum() for a in authors], dtype=float)        # A
    a_idx = np.array([authors.index(v) for v in y])
    raw = np.broadcast_to(sums, (n,) + sums.shape).copy()                # n x A x k
    ncnt = np.broadcast_to(cnt, (n, len(authors))).copy()
    raw[np.arange(n), a_idx] -= X
    ncnt[np.arange(n), a_idx] -= 1
    C = (raw / ncnt[..., None] - mu[:, None, :]) / sd[:, None, :]
    z = (X - mu) / sd
    burrows = np.abs(z[:, None, :] - C).mean(axis=2)
    cos = 1 - (C @ z[..., None])[..., 0] / (np.linalg.norm(C, axis=2) * np.linalg.norm(z, axis=1)[:, None])
    return {"burrows": burrows, "cosine": cos}


# ── PCA ───────────────────────────────────────────────────────────────────────

@dataclass
class PCAModel:
    authors: list[str]
    cfg: Config
    keep: np.ndarray = field(init=False)
    info: dict = field(init=False, default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray, col_mask: np.ndarray) -> "PCAModel":
        cols = np.where(col_mask)[0]
        Xs = X[:, cols]
        nan_ratio = np.isnan(Xs).mean(axis=0)
        ok_nan = nan_ratio <= self.cfg.max_nan_ratio
        self.means = np.nanmean(Xs, axis=0)
        Xi = np.where(np.isnan(Xs), self.means, Xs)
        ok_var = Xi.std(axis=0) > 0
        keep_local = ok_nan & ok_var
        self.keep, self.means = cols[keep_local], self.means[keep_local]
        Xi = Xi[:, keep_local]
        self.scaler = StandardScaler().fit(Xi)
        self.pca = PCA(n_components=self.cfg.pca_variance, svd_solver="full").fit(self.scaler.transform(Xi))
        P = self.pca.transform(self.scaler.transform(Xi))
        self.C = np.vstack([P[y == a].mean(axis=0) for a in self.authors])
        self.P_ref = P
        self.info = {"n_input": int(len(cols)), "dropped_nan": int((~ok_nan).sum()),
                     "dropped_zero_var": int((ok_nan & ~ok_var).sum()), "n_used": int(len(self.keep)),
                     "n_components": int(self.pca.n_components_),
                     "variance_explained": float(self.pca.explained_variance_ratio_.sum())}
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        Xs = X[:, self.keep]
        Xs = np.where(np.isnan(Xs), self.means, Xs)
        return self.pca.transform(self.scaler.transform(Xs))

    def distances(self, X: np.ndarray) -> np.ndarray:
        P = self.project(X)
        return cdist(P, self.C, metric="cityblock") / P.shape[1]

    def shortcut_loo_own(self, y: np.ndarray) -> np.ndarray:
        """Distance of each reference chunk to its own centroid with that chunk removed
        (scaler and PCA kept fixed). Used for in-fold thresholds in LOWO."""
        out = np.empty(len(y))
        for j, a in enumerate(self.authors):
            m = y == a
            n_a = m.sum()
            c_loo = (n_a * self.C[j] - self.P_ref[m]) / (n_a - 1)
            out[m] = np.abs(self.P_ref[m] - c_loo).sum(axis=1) / self.P_ref.shape[1]
        return out


# ── Bundled space ─────────────────────────────────────────────────────────────

class Space:
    """Vocabulary + Delta + PCA levels fitted on one reference set."""

    def __init__(self, corpus: Corpus, ref_idx: np.ndarray, y: np.ndarray, cfg: Config):
        self.corpus, self.cfg, self.authors = corpus, cfg, list(cfg.authors)
        self.vocab = select_vocab(corpus, ref_idx, cfg)
        self.columns, groups = self.vocab.columns()
        self.groups = np.array(groups)
        self.fw_mask = self.groups == "fw"
        self.X_ref, self.y = feature_matrix(corpus, ref_idx, self.vocab), y
        self.delta = DeltaModel(self.authors).fit(self.X_ref[:, self.fw_mask], y)
        self.pca = {name: PCAModel(self.authors, cfg).fit(self.X_ref, y, self.level_mask(name))
                    for name in cfg.pca_levels}

    def level_mask(self, level: str) -> np.ndarray:
        return ~np.isin(self.groups, list(self.cfg.pca_levels[level]))

    def matrix(self, idx: np.ndarray) -> np.ndarray:
        return feature_matrix(self.corpus, idx, self.vocab)

    def distances(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out = self.delta.distances(X[:, self.fw_mask])
        for name, model in self.pca.items():
            out[name] = model.distances(X)
        return out

    def feature_counts(self) -> list[dict]:
        rows = []
        for name, model in self.pca.items():
            mask = self.level_mask(name)
            row = {"level": name}
            for g in ("other", "fw", "wngram", "pos"):
                row[f"n_{g}"] = int((mask & (self.groups == g)).sum())
            rows.append({**row, **model.info})
        rows.append({"level": "delta", "n_fw": int(self.fw_mask.sum())})
        return rows


def decisions(dist: dict[str, np.ndarray], own: list[str], authors: list[str],
              thresholds: dict | None = None) -> pd.DataFrame:
    """Long table: one row per (document, method). `own` = target/true author ('' if none)."""
    rows = []
    for method, D in dist.items():
        nearest = np.array(authors)[D.argmin(axis=1)]
        for r in range(len(D)):
            row = {"method": method, "row": r, "nearest": nearest[r]}
            a = own[r]
            if a:
                j = authors.index(a)
                row["d_own"] = float(D[r, j])
                row["rank_own"] = int((D[r] < D[r, j]).sum()) + 1
                row["attr_fail"] = bool(nearest[r] != a)
                if thresholds is not None:
                    thr = thresholds[method][a]
                    row["threshold"] = thr
                    row["verif_reject"] = bool(D[r, j] > thr)
            for j, b in enumerate(authors):
                row[f"d_{b}"] = float(D[r, j])
            rows.append(row)
    return pd.DataFrame(rows)
