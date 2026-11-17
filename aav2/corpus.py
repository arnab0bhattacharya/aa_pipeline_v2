"""Stage `prepare`: assemble every text the study uses into one document table.

Sets:
  reference  140 chunks/author  (fit set for every model)
  anchor      25 chunks/author  (never-seen authentic chunks; the prompts' style exemplars)
  spare      top-up chunks for leave-one-work-out folds (only the ones a fold needs)
  imitation  2,000 generations (target author known)
  control      500 generations (no author named; linked to a scenario's author)

Outputs (out/):
  documents.csv         one row per document, with the path of the text actually analysed
  lowo_plan.csv         one row per fold: held-out work and the spare chunks that top it up
  prepared_generated/   generated texts after truncation to ~1,000 words
"""
from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import Config
from .textutil import detect_paragraphs, extract_first_chunk, normalise_text


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _authentic_rows(cfg: Config) -> list[dict]:
    rows = []
    ref_dir = cfg.path(cfg.reference_dir)
    for r in _read_csv(ref_dir / "chunk_manifest.csv"):
        rows.append({"doc_id": r["chunk_id"], "set": "reference", "author": r["author"],
                     "work": r["work"], "path": str(ref_dir / r["author"] / f"{r['chunk_id']}.txt")})
    anc_dir = cfg.path(cfg.anchor_dir)
    for r in _read_csv(anc_dir / "anchor_manifest.csv"):
        rows.append({"doc_id": r["chunk_id"], "set": "anchor", "author": r["author"],
                     "work": r["work"], "anchor_id": r["anchor_id"],
                     "path": str(anc_dir / r["author"] / f"{r['chunk_id']}.txt")})
    return rows


def _lowo_plan(cfg: Config, reference: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """One fold per (author, work) in the reference set.

    The held-out work's reference chunks become the fold's test set. The author is
    topped back up to 140 with spare chunks from the author's *other* works, drawn
    uniformly at random (seeded). If too few spares exist (Wilde), the fold runs short
    and the shortfall is recorded.
    """
    rng = random.Random(cfg.seed)
    spares = _read_csv(cfg.path(cfg.spare_manifest))
    by_author: dict[str, list[dict]] = {}
    for s in spares:
        by_author.setdefault(s["author"], []).append(s)

    folds, needed = [], {}
    for (author, work), grp in reference.groupby(["author", "work"], sort=True):
        n_out = len(grp)
        pool = sorted((s for s in by_author.get(author, []) if s["work"] != work),
                      key=lambda s: s["chunk_id"])
        topup = rng.sample(pool, min(n_out, len(pool)))
        for s in topup:
            needed[s["chunk_id"]] = s
        folds.append({"fold": f"{author}|{work}", "author": author, "work": work,
                      "n_test": n_out, "n_topup": len(topup), "shortfall": n_out - len(topup),
                      "topup_ids": ";".join(s["chunk_id"] for s in topup)})

    spare_dir = cfg.path(cfg.spare_dir)
    spare_rows = [{"doc_id": s["chunk_id"], "set": "spare", "author": s["author"], "work": s["work"],
                   "path": str(spare_dir / s["author"] / f"{s['chunk_id']}.txt")}
                  for s in sorted(needed.values(), key=lambda s: s["chunk_id"])]
    return folds, spare_rows


def _generated_rows(cfg: Config) -> list[dict]:
    calls = _read_csv(cfg.path(cfg.call_manifest))
    control_key = {r["prompt_id"]: r for r in _read_csv(cfg.path(cfg.control_key))}
    anchor_of = {r["scenario_id"]: r["anchor_id"] for r in _read_csv(cfg.path(cfg.scenarios))}

    main_dir, regen_dir = cfg.path(cfg.generated_dir), cfg.path(cfg.regenerated_dir)
    prep_dir = cfg.out / "prepared_generated"
    prep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in calls:
        call_id = c["call_id"]
        src = regen_dir / f"{call_id}.txt"
        regenerated = src.exists()
        if not regenerated:
            src = main_dir / f"{call_id}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Generated text missing: {src}")

        raw = normalise_text(src.read_text(encoding="utf-8"))
        paras = detect_paragraphs(raw, cfg.min_para_density)
        text = extract_first_chunk(paras, cfg.chunk_target_words, cfg.chunk_window, cfg.sentence_end_regex)
        out_path = prep_dir / f"{call_id}.txt"
        out_path.write_text(text, encoding="utf-8")

        if c["arm"] == "imitation":
            target, scenario = c["author"], c["scenario_id"]
        elif c["arm"] == "control":
            key = control_key[c["prompt_id"]]
            target, scenario = "", key["scenario_id"]
        else:
            raise ValueError(f"Unknown arm {c['arm']!r} for {call_id}")

        rows.append({
            "doc_id": call_id, "set": c["arm"], "author": target,
            "scenario_author": scenario.split("_")[0], "scenario_id": scenario,
            "anchor_id": anchor_of[scenario], "model_slot": c["model_slot"],
            "regenerated": regenerated, "raw_words": len(raw.split()),
            "path": str(out_path),
        })
    return rows


def run(cfg: Config) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    authentic = pd.DataFrame(_authentic_rows(cfg))
    reference = authentic[authentic["set"] == "reference"]

    counts = Counter(reference["author"])
    _check(sorted(counts) == sorted(cfg.authors), f"Reference authors {sorted(counts)} != design")
    _check(all(v == cfg.reference_per_author for v in counts.values()), f"Reference counts {dict(counts)}")
    anchors = authentic[authentic["set"] == "anchor"]
    a_counts = Counter(anchors["author"])
    _check(all(a_counts[a] == cfg.anchors_per_author for a in cfg.authors), f"Anchor counts {dict(a_counts)}")
    _check(not set(reference["doc_id"]) & set(anchors["doc_id"]), "Anchor overlaps reference")

    folds, spare_rows = _lowo_plan(cfg, reference)
    _check(not {r["doc_id"] for r in spare_rows} & set(authentic["doc_id"]), "Spare overlaps reference/anchor")

    gen = pd.DataFrame(_generated_rows(cfg))
    _check((gen["set"] == "imitation").sum() == cfg.n_imitations, "Imitation count != design")
    _check((gen["set"] == "control").sum() == cfg.n_controls, "Control count != design")

    docs = pd.concat([authentic, pd.DataFrame(spare_rows), gen], ignore_index=True)
    _check(docs["doc_id"].is_unique, "Duplicate doc_id across sets")
    for p in docs["path"]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    docs.to_csv(cfg.out / "documents.csv", index=False)
    pd.DataFrame(folds).to_csv(cfg.out / "lowo_plan.csv", index=False)
    print(f"prepare: {docs['set'].value_counts().to_dict()}; "
          f"{len(folds)} LOWO folds, {int(gen['regenerated'].sum())} regenerated texts used")
