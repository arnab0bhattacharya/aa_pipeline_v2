"""Stages `validate` and `score`.

validate (authentic text only; gives the false-positive side of every comparison)
  LOO      every reference chunk left out in turn. Delta: exact refit of z-statistics
           and centroids. PCA: scaler, PCA and centroids refitted without the chunk
           (vocabulary held fixed). Each held-out chunk's distance to its own author
           is that author's calibration set.
  thresholds  per method x author: the 95th percentile of the LOO own-author distances.
           A text farther than this from the target is rejected (verification).
  anchors  250 never-seen authentic chunks scored against the full reference model.
  LOWO     one fold per (author, work): the work's reference chunks are held out,
           the author is topped up from spare chunks of other works, and vocabulary,
           Delta and PCA are all refitted. In-fold thresholds for the held-out author
           come from a shortcut LOO inside the fold (Delta exact; PCA with the fold's
           scaler/PCA fixed and only the centroid adjusted).

score
  All 2,500 generated texts against the full reference model and the LOO thresholds.
  Imitations are judged against their target; controls carry distances to all ten
  authors and, for each author, whether that author's threshold would accept them.

Outputs (out/): feature_counts.csv, vocabulary.json, loo.csv, thresholds.csv,
anchors.csv, lowo.csv, generated.csv
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .space import Corpus, PCAModel, Space, decisions, delta_loo


def load_corpus(cfg: Config) -> tuple[pd.DataFrame, Corpus]:
    docs = pd.read_csv(cfg.out / "documents.csv", keep_default_na=False)
    with open(cfg.out / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    fixed = pd.read_csv(cfg.out / "fixed_features.csv")
    return docs, Corpus(docs, sequences, fixed)


def full_space(cfg: Config, docs: pd.DataFrame, corpus: Corpus) -> tuple[Space, pd.DataFrame]:
    ref = docs[docs["set"] == "reference"]
    return Space(corpus, corpus.idx(ref["doc_id"]), ref["author"].to_numpy(), cfg), ref


def _attach(meta: pd.DataFrame, dec: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = meta[cols].reset_index(drop=True)
    return dec.join(m, on="row").drop(columns="row")[cols + [c for c in dec.columns if c not in cols + ["row"]]]


def run_validate(cfg: Config) -> None:
    docs, corpus = load_corpus(cfg)
    space, ref = full_space(cfg, docs, corpus)
    authors = list(cfg.authors)
    y = space.y

    pd.DataFrame(space.feature_counts()).to_csv(cfg.out / "feature_counts.csv", index=False)
    with open(cfg.out / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(space.vocab.names, f, indent=1)

    # ── LOO ──────────────────────────────────────────────────────────────────
    dist = delta_loo(space.X_ref[:, space.fw_mask], y, authors)
    for level in cfg.pca_levels:
        mask = space.level_mask(level)
        D = np.empty((len(y), len(authors)))
        for i in tqdm(range(len(y)), desc=f"LOO {level}", unit="chunk"):
            keep = np.arange(len(y)) != i
            model = PCAModel(authors, cfg).fit(space.X_ref[keep], y[keep], mask)
            D[i] = model.distances(space.X_ref[i: i + 1])[0]
        dist[level] = D
    loo = _attach(ref, decisions(dist, list(y), authors), ["doc_id", "author", "work"])
    loo.to_csv(cfg.out / "loo.csv", index=False)

    thr = (loo.groupby(["method", "author"])["d_own"]
              .quantile(cfg.threshold_percentile / 100).rename("threshold").reset_index())
    thr.to_csv(cfg.out / "thresholds.csv", index=False)
    thresholds = {m: dict(zip(g["author"], g["threshold"])) for m, g in thr.groupby("method")}

    # ── Anchors ──────────────────────────────────────────────────────────────
    anc = docs[docs["set"] == "anchor"]
    D = space.distances(space.matrix(corpus.idx(anc["doc_id"])))
    _attach(anc, decisions(D, list(anc["author"]), authors, thresholds),
            ["doc_id", "author", "work", "anchor_id"]).to_csv(cfg.out / "anchors.csv", index=False)

    # ── LOWO ─────────────────────────────────────────────────────────────────
    plan = pd.read_csv(cfg.out / "lowo_plan.csv", keep_default_na=False)
    out = []
    for fold in tqdm(plan.itertuples(), total=len(plan), desc="LOWO", unit="fold"):
        held = ref[(ref["author"] == fold.author) & (ref["work"] == fold.work)]
        train = ref[~ref["doc_id"].isin(held["doc_id"])]
        topup = [t for t in fold.topup_ids.split(";") if t]
        train_ids = list(train["doc_id"]) + topup
        y_tr = np.concatenate([train["author"].to_numpy(), np.full(len(topup), fold.author)])
        fs = Space(corpus, corpus.idx(train_ids), y_tr, cfg)

        # in-fold thresholds for the held-out author
        own = {}
        d_loo = delta_loo(fs.X_ref[:, fs.fw_mask], y_tr, authors)
        a_rows = y_tr == fold.author
        j = authors.index(fold.author)
        for m in cfg.delta_methods:
            own[m] = d_loo[m][a_rows, j]
        for level, model in fs.pca.items():
            own[level] = model.shortcut_loo_own(y_tr)[a_rows]
        fold_thr = {m: {fold.author: float(np.percentile(v, cfg.threshold_percentile))} for m, v in own.items()}

        D = fs.distances(fs.matrix(corpus.idx(held["doc_id"])))
        dec = _attach(held, decisions(D, list(held["author"]), authors, fold_thr), ["doc_id", "author", "work"])
        dec.insert(0, "fold", fold.fold)
        dec["n_train_author"] = int(a_rows.sum())
        dec["n_mfw"] = int(fs.fw_mask.sum())
        out.append(dec)
    pd.concat(out, ignore_index=True).to_csv(cfg.out / "lowo.csv", index=False)
    print(f"validate: LOO {len(y)} chunks, {len(anc)} anchors, {len(plan)} LOWO folds")


def run_score(cfg: Config) -> None:
    docs, corpus = load_corpus(cfg)
    space, _ = full_space(cfg, docs, corpus)
    authors = list(cfg.authors)
    thr = pd.read_csv(cfg.out / "thresholds.csv")
    thresholds = {m: dict(zip(g["author"], g["threshold"])) for m, g in thr.groupby("method")}

    gen = docs[docs["set"].isin(["imitation", "control"])]
    D = space.distances(space.matrix(corpus.idx(gen["doc_id"])))
    dec = _attach(gen, decisions(D, list(gen["author"]), authors, thresholds),
                  ["doc_id", "set", "author", "scenario_author", "scenario_id", "anchor_id", "model_slot", "raw_words"])
    for a in authors:  # would author a's threshold accept this text?
        dec[f"accept_{a}"] = dec[f"d_{a}"] <= dec["method"].map(lambda m: thresholds[m][a])
    dec.to_csv(cfg.out / "generated.csv", index=False)
    print(f"score: {len(gen)} generated texts x {len(D)} methods")
