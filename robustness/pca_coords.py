"""Export full-PCA coordinates for the PC1 x PC2 figure.

Fits the same full reference space as the `score` stage (aav2.evaluate.full_space) and
projects the 1,400 reference chunks and the 2,500 generated texts into the pca_full space.
Only the first two components are written; the diagnostic itself uses all k* components.

Run:   .venv\\Scripts\\python robustness\\pca_coords.py
Writes: out/results/pca_coords.csv   (doc_id, set, author, model_slot, PC1, PC2)
        out/results/pca_variance.csv (component, explained_variance_ratio) for the first 10 PCs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aav2.config import Config  # noqa: E402
from aav2.evaluate import full_space, load_corpus  # noqa: E402


def main():
    cfg = Config()
    docs, corpus = load_corpus(cfg)
    space, ref = full_space(cfg, docs, corpus)
    pca = space.pca["pca_full"]

    P_ref = pca.P_ref[:, :2]
    gen = docs[docs["set"].isin(["imitation", "control"])]
    P_gen = pca.project(space.matrix(corpus.idx(gen["doc_id"])))[:, :2]

    # sanity: projecting the reference rows again reproduces P_ref
    again = pca.project(space.X_ref)[:, :2]
    assert np.allclose(again, P_ref), "projection does not reproduce fitted reference coordinates"

    out = pd.concat([
        pd.DataFrame({"doc_id": ref["doc_id"].to_numpy(), "set": "reference", "author": ref["author"].to_numpy(),
                      "model_slot": "", "PC1": P_ref[:, 0], "PC2": P_ref[:, 1]}),
        pd.DataFrame({"doc_id": gen["doc_id"].to_numpy(), "set": gen["set"].to_numpy(),
                      "author": np.where(gen["set"] == "imitation", gen["author"], gen["scenario_author"]),
                      "model_slot": gen["model_slot"].to_numpy(), "PC1": P_gen[:, 0], "PC2": P_gen[:, 1]}),
    ])
    out.to_csv(cfg.out / "results" / "pca_coords.csv", index=False)
    evr = pca.pca.explained_variance_ratio_
    pd.DataFrame({"component": np.arange(1, 11), "explained_variance_ratio": evr[:10]}).to_csv(
        cfg.out / "results" / "pca_variance.csv", index=False)
    print(f"k* = {pca.pca.n_components_}, total = {evr.sum():.4f}, PC1 = {evr[0]:.4f}, PC2 = {evr[1]:.4f}")
    print(out.groupby("set").size().to_dict())
    print(out[out.set == "reference"].groupby("author")[["PC1", "PC2"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
