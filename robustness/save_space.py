"""Save the fitted reference spaces (Delta + the three PCA levels) to disk.

This calls the unchanged load_corpus() + full_space() that the
`score` stage uses, then writes every fitted parameter and every text's coordinates, and checks
that distances recomputed from the saved arrays reproduce out/generated.csv exactly.

Run:    .venv\\Scripts\\python robustness\\save_space.py
Writes: out/space/
          features.csv                    column name + feature group for all 282 features
          X_{reference,anchor,generated}.npy   raw feature matrices (rows match docs_*.csv)
          docs_{reference,anchor,generated}.csv
          delta.npz                       mfw columns, mu, sd, author signatures C
          pca_{full,no_fw,no_lex}.npz     keep idx, impute means, scaler mean/scale, components,
                                          explained variance (+ratio), centroids C,
                                          scores for reference / anchors / generated (all k* PCs)
          pca_{level}_loadings.csv        loadings (component x sqrt(eigenvalue)) for PC1-PC5
          pca_{level}_centroids.csv       author centroids on PC1-PC5
          manifest.json                   versions, shapes, k*, variance, verification result
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aav2.config import Config  # noqa: E402
from aav2.evaluate import full_space, load_corpus  # noqa: E402


def main():
    cfg = Config()
    out = cfg.out / "space"
    out.mkdir(exist_ok=True)
    docs, corpus = load_corpus(cfg)
    space, ref = full_space(cfg, docs, corpus)
    authors = list(cfg.authors)

    sets = {"reference": ref, "anchor": docs[docs["set"] == "anchor"],
            "generated": docs[docs["set"].isin(["imitation", "control"])]}
    X = {"reference": space.X_ref}
    X["anchor"] = space.matrix(corpus.idx(sets["anchor"]["doc_id"]))
    X["generated"] = space.matrix(corpus.idx(sets["generated"]["doc_id"]))
    for k, d in sets.items():
        np.save(out / f"X_{k}.npy", X[k])
        d.drop(columns=["path"], errors="ignore").to_csv(out / f"docs_{k}.csv", index=False)
    pd.DataFrame({"feature": space.columns, "group": space.groups}).to_csv(out / "features.csv", index=False)

    d = space.delta
    np.savez(out / "delta.npz", mfw=np.array(space.columns)[space.fw_mask], mu=d.mu, sd=d.sd, C=d.C,
             authors=np.array(authors))

    manifest = {"authors": authors, "n_features": len(space.columns), "levels": {}}
    for level, p in space.pca.items():
        scores = {k: p.project(X[k]) for k in X}
        assert np.allclose(scores["reference"], p.P_ref), level
        np.savez(out / f"{level}.npz", keep=p.keep, impute_means=p.means, scaler_mean=p.scaler.mean_,
                 scaler_scale=p.scaler.scale_, pca_mean=p.pca.mean_, components=p.pca.components_,
                 explained_variance=p.pca.explained_variance_, explained_variance_ratio=p.pca.explained_variance_ratio_,
                 C=p.C, scores_reference=scores["reference"], scores_anchor=scores["anchor"],
                 scores_generated=scores["generated"], authors=np.array(authors))
        cols = np.array(space.columns)[p.keep]
        L = p.pca.components_[:5].T * np.sqrt(p.pca.explained_variance_[:5])
        pd.DataFrame(L, index=cols, columns=[f"PC{i}" for i in range(1, 6)]).to_csv(out / f"{level}_loadings.csv")
        pd.DataFrame(p.C[:, :5], index=authors, columns=[f"PC{i}" for i in range(1, 6)]).to_csv(
            out / f"{level}_centroids.csv")
        manifest["levels"][level] = {**p.info, "pc_variance_ratio_1to5": p.pca.explained_variance_ratio_[:5].tolist()}

    # verification: recompute every distance from the SAVED arrays only, compare with generated.csv
    gen = pd.read_csv(cfg.out / "generated.csv")
    Xg = np.load(out / "X_generated.npy")
    ids = pd.read_csv(out / "docs_generated.csv")["doc_id"]
    worst = {}
    dz = np.load(out / "delta.npz")
    fw = pd.read_csv(out / "features.csv")["group"].eq("fw").to_numpy()
    Z = (Xg[:, fw] - dz["mu"]) / dz["sd"]
    mine = {"burrows": np.abs(Z[:, None, :] - dz["C"][None]).mean(-1),
            "cosine": 1 - (Z @ dz["C"].T) / (np.linalg.norm(Z, axis=1)[:, None] * np.linalg.norm(dz["C"], axis=1)[None])}
    for level in cfg.pca_levels:
        z = np.load(out / f"{level}.npz")
        Xs = Xg[:, z["keep"]]
        Xs = np.where(np.isnan(Xs), z["impute_means"], Xs)
        S = ((Xs - z["scaler_mean"]) / z["scaler_scale"] - z["pca_mean"]) @ z["components"].T
        mine[level] = np.abs(S[:, None, :] - z["C"][None]).sum(-1) / S.shape[1]
    for m, D in mine.items():
        g = gen[gen["method"] == m].set_index("doc_id").loc[ids, [f"d_{a}" for a in authors]].to_numpy()
        worst[m] = float(np.nanmax(np.abs(D - g)))
    manifest["verification_max_abs_diff_vs_generated_csv"] = worst
    import sklearn, spacy
    manifest["versions"] = {"numpy": np.__version__, "sklearn": sklearn.__version__, "spacy": spacy.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["levels"], indent=1))
    print("max |saved-space distance - generated.csv|:", worst)


if __name__ == "__main__":
    main()
