"""All pipeline parameters and input/output locations. Single source of truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_DIR.parent  # Repository root (aa_pipeline_v2)


@dataclass(frozen=True)
class Config:
    root: Path = DEFAULT_ROOT
    out: Path = field(default_factory=lambda: DEFAULT_ROOT / "out")

    # ── Inputs (relative to root) ──────────────────────────────────────────
    reference_dir: str = "3_sampled_chunked_books_140"
    anchor_dir: str = "anchors"
    spare_manifest: str = "anchors/spare_manifest.csv"
    spare_dir: str = "2_chunked_books"
    generated_dir: str = "generation/outputs/main/texts"
    # Regenerated texts placed here replace the same call_id in generated_dir.
    regenerated_dir: str = "generation/outputs/regen/texts"
    call_manifest: str = "prompts/call_manifest.csv"
    control_key: str = "prompts/control_key.csv"
    scenarios: str = "scenarios/scenarios_final_paired.csv"

    # ── Expected design counts (asserted) ──────────────────────────────────
    authors: tuple[str, ...] = ("ACD", "CB", "CD", "GE", "HGW", "JA", "JC", "MS", "OW", "VW")
    reference_per_author: int = 140
    anchors_per_author: int = 25
    n_imitations: int = 2000
    n_controls: int = 500
    model_slots: tuple[str, ...] = ("M1", "M2", "M3", "M4")

    # ── Generated text truncation (same rule as the reference chunker) ─────
    chunk_target_words: int = 1000
    chunk_window: int = 0
    min_chunk_words: int = 1000
    min_para_density: int = 2
    sentence_end_regex: str = r"[.!?]$"

    # ── Vocabulary (refit on every reference set: full, each LOO/LOWO fold) ─
    token_regex: str = r"\b[a-z]+\b"
    min_doc_freq_ratio: float = 0.5   # MFW = every word in >= 50% of reference chunks
    max_mfw: int = 500                # safety cap only; the doc-frequency rule binds
    num_word_bigrams: int = 100
    num_word_trigrams: int = 50
    num_pos_bigrams: int = 50
    num_pos_trigrams: int = 30

    # ── PCA ────────────────────────────────────────────────────────────────
    pca_variance: float = 0.95
    max_nan_ratio: float = 0.05
    # Ablation levels: feature groups removed from the PCA input.
    pca_levels: dict = field(default_factory=lambda: {
        "pca_full": (),
        "pca_no_fw": ("fw",),
        "pca_no_lex": ("fw", "wngram"),
    })

    # ── Verification ───────────────────────────────────────────────────────
    threshold_percentile: float = 95.0

    # ── Statistics ─────────────────────────────────────────────────────────
    seed: int = 42
    bootstrap: int = 5000
    equivalence_margin: float = 0.10
    mfw_sweep: tuple[int, ...] = (50, 100)   # plus "all MFW" appended at run time

    def path(self, rel: str) -> Path:
        return self.root / rel

    @property
    def delta_methods(self) -> tuple[str, ...]:
        return ("burrows", "cosine")

    @property
    def methods(self) -> tuple[str, ...]:
        return self.delta_methods + tuple(self.pca_levels)
