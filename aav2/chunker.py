"""Segment standardized source books into standardized chunks (~1,000 words).

Processes plain text novels from `1_standardized_books` into standardized chunks in
`2_chunked_books`, preserving paragraph structure and generating `chunk_manifest.csv`.

Inputs:
  1_standardized_books/*.txt    Normalized full-text novel files (AUTHOR_Work_Title.txt)

Outputs:
  2_chunked_books/{AUTHOR}/{chunk_id}.txt
  2_chunked_books/chunk_manifest.csv

Usage:
  python -m aav2.chunker
  python -m aav2.chunker --input 1_standardized_books --output 2_chunked_books
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from .config import Config
from .textutil import chunk_paragraphs, detect_paragraphs, normalise_text


def parse_filename(stem: str, suffix_regex: str | None = None) -> tuple[str, str]:
    """Extract (author, work) from filename stems.

    Expected format: AUTHOR_Work_Title_standardised or AUTHOR_Work_Title.
    """
    pattern = suffix_regex if suffix_regex is not None else r"_standardis[e]?d$"
    stem_clean = re.sub(pattern, "", stem, flags=re.IGNORECASE)
    if "_" in stem_clean:
        author, work = stem_clean.split("_", 1)
        return author.strip(), work.strip()
    return "Unknown", stem_clean.strip()


def write_manifest(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write chunk manifest rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)


def run(
    cfg: Config | None = None,
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Segment all standardized books into ~chunk_target_words chunks."""
    if cfg is None:
        cfg = Config()

    in_dir = Path(input_dir) if input_dir else cfg.path(cfg.raw_books_dir)
    out_dir = Path(output_dir) if output_dir else cfg.path(cfg.chunked_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory '{in_dir}' not found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in '{in_dir}'.")

    target_size = cfg.chunk_target_words
    chunk_window = cfg.chunk_window
    min_chunk_size = cfg.min_chunk_words
    min_para_density = cfg.min_para_density
    suffix_pattern = cfg.standardised_suffix_regex
    sentence_end_pat = cfg.sentence_end_regex

    manifest_rows: list[dict] = []
    author_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    total_chunks = 0

    for filepath in files:
        author, work = parse_filename(filepath.stem, suffix_regex=suffix_pattern)

        raw_text = filepath.read_text(encoding="utf-8", errors="replace")
        text = normalise_text(raw_text)
        paras = detect_paragraphs(text, min_density=min_para_density)
        chunks = chunk_paragraphs(
            paras,
            target=target_size,
            window=chunk_window,
            min_size=min_chunk_size,
            sentence_end_re=sentence_end_pat,
        )

        if not chunks:
            continue

        author_dir = out_dir / author
        author_dir.mkdir(parents=True, exist_ok=True)

        for chunk_idx, (chunk_text, boundary_type) in enumerate(chunks, start=1):
            chunk_id = f"{author}_{work}_chunk_{chunk_idx}"
            chunk_path = author_dir / f"{chunk_id}.txt"
            chunk_path.write_text(chunk_text, encoding="utf-8")

            manifest_rows.append({
                "chunk_id": chunk_id,
                "author": author,
                "work": work,
                "chunk_number": chunk_idx,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "source_file": filepath.name,
                "boundary_type": boundary_type,
                "is_reference": True,
            })

            author_counts[author] += 1
            boundary_counts[boundary_type] += 1
            total_chunks += 1

    manifest_path = out_dir / "chunk_manifest.csv"
    manifest_fields = [
        "chunk_id", "author", "work", "chunk_number",
        "word_count", "char_count", "source_file",
        "boundary_type", "is_reference",
    ]
    write_manifest(manifest_path, manifest_rows, manifest_fields)

    print(
        f"chunker: Segmented {len(files)} books into {total_chunks} chunks across "
        f"{len(author_counts)} authors. Manifest written to {manifest_path}"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk standardized books into ~1,000 word units.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=None,
        help="Input directory containing standardized books (default: cfg.raw_books_dir)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory for chunks and manifest (default: cfg.chunked_dir)",
    )
    args = parser.parse_args()
    cfg = Config()
    run(cfg, input_dir=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
