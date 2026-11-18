"""python -m aav2 <stage> [--out DIR]

Stages run in this order; `all` runs the core pipeline stages (prepare through analyze).
  chunk     segment 1_standardized_books into 2_chunked_books
  prepare   assemble documents, truncate generated texts, plan LOWO folds
  extract   one spaCy parse per document; fixed features + token/POS cache
  validate  LOO, thresholds, anchors, LOWO (authentic text)
  score     generated imitations and controls
  analyze   contingency, model contrasts, control dynamics, MFW sensitivity
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from . import analysis, chunker, corpus, evaluate, features
from .config import Config

PIPELINE_STAGES = ("prepare", "extract", "validate", "score", "analyze")
STAGES = {
    "chunk": chunker.run,
    "prepare": corpus.run,
    "extract": features.run,
    "validate": evaluate.run_validate,
    "score": evaluate.run_score,
    "analyze": analysis.run,
}


def main() -> None:
    p = argparse.ArgumentParser(prog="aav2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(STAGES) + ["all"])
    p.add_argument("--out", type=Path, help="output directory (default: aa_pipeline_v2/out)")
    a = p.parse_args()
    cfg = Config() if a.out is None else replace(Config(), out=a.out.resolve())
    for name in (PIPELINE_STAGES if a.stage == "all" else [a.stage]):
        t0 = time.time()
        STAGES[name](cfg)
        print(f"[{name}] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
