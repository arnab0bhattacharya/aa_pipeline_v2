"""Draw anchors from the chunks left over after the 140-per-author reference
sample, and pair them with the final scenarios.

  1. Leftover pool = reference chunks in 2_chunked_books not in
     3_sampled_chunked_books_140.
  2. 25 anchors per author, stratified across works by leftover pool size,
     using the pipeline's own stratified_sample (Hamilton + cap redistribution).
  3. Each anchor is paired with exactly 2 of the author's 50 scenarios by a
     seeded shuffle; the pairing is shared by all models.

Outputs (in ../anchors/):
  anchor_manifest.csv            one row per anchor
  {AUTHOR}/{chunk_id}.txt        anchor texts
  spare_manifest.csv             leftover chunks that are neither reference nor anchor
And scenarios_final_paired.csv (scenarios + anchor_id) in this folder.

Usage: .venv/Scripts/python build_anchors.py
"""
import csv
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
ANCHORS_PER_AUTHOR = 25
USES_PER_ANCHOR = 2
CHUNKED = ROOT / "2_chunked_books"
SAMPLED = ROOT / "3_sampled_chunked_books_140"
OUT = ROOT / "anchors"

# Narrative form of each reference work (for later anchor-form checks)
STORY_WORKS = {
    "ACD": {"The_Adventures_of_Sherlock_Holmes", "The_Return_of_Sherlock_Holmes"},
    "MS": {"The_Heir_of_Mondolfo"},
    "OW": {"Lord_Arthur_Savile's_Crime_The_Portrait_of_Mr._W.H_and_Other_Stories"},
    "VW": {"Monday_or_Tuesday"},
}
TALE_WORKS = {"OW": {"A_House_of_Pomegranates", "The_Happy_Prince_and_Other_Tales"}}


def load_manifest(path: Path | str, filter_reference: bool | None = None) -> list[dict]:
    """Load a CSV manifest file, optionally filtering by reference status."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    rows = []
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filter_reference is not None:
                is_ref = row.get("is_reference", "").strip().lower() in ("true", "1", "yes")
                if is_ref != filter_reference:
                    continue
            rows.append(row)
    return rows


def write_manifest(path: Path | str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write rows to a CSV manifest."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)


def hamilton_allocate(weights: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` integer slots proportionally across keys using the
    Hamilton / Largest Remainder method.
    """
    if not weights or total == 0:
        return {k: 0 for k in weights}
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return {k: 0 for k in weights}
    raw = {k: (v / weight_sum) * total for k, v in weights.items()}
    floors = {k: math.floor(v) for k, v in raw.items()}
    remainders = {k: raw[k] - floors[k] for k in weights}
    leftover = total - sum(floors.values())
    sorted_keys = sorted(weights.keys(), key=lambda k: remainders[k], reverse=True)
    result = dict(floors)
    for k in sorted_keys[:leftover]:
        result[k] += 1
    return result


def stratified_sample(
    chunks_by_book: dict[str, list],
    target: int,
    rng: random.Random,
) -> list:
    """Return exactly `target` items drawn proportionally from each book,
    never exceeding a book's available pool.
    """
    books = list(chunks_by_book.keys())
    pool_sizes = {b: len(chunks_by_book[b]) for b in books}
    total_pool = sum(pool_sizes.values())

    if total_pool <= target:
        return [f for files in chunks_by_book.values() for f in files]

    # Step 1: initial Hamilton allocation
    quotas = hamilton_allocate(pool_sizes, target)

    # Steps 2-4: iterative redistribution for capped books
    while True:
        capped = {b for b in books if quotas[b] > pool_sizes[b]}
        if not capped:
            break

        surplus = sum(quotas[b] - pool_sizes[b] for b in capped)
        for b in capped:
            quotas[b] = pool_sizes[b]

        headroom = {
            b: pool_sizes[b] - quotas[b]
            for b in books
            if b not in capped and pool_sizes[b] > quotas[b]
        }

        if not headroom or surplus == 0:
            break

        extra = hamilton_allocate(headroom, min(surplus, sum(headroom.values())))
        for b, e in extra.items():
            quotas[b] += e

    # Sample
    sampled = []
    for b in books:
        q = min(quotas[b], pool_sizes[b])
        if q > 0:
            sampled.extend(rng.sample(chunks_by_book[b], q))

    return sampled


def form_of(author: str, work: str) -> str:
    if work in TALE_WORKS.get(author, set()):
        return "fairy-tale collection"
    if work in STORY_WORKS.get(author, set()):
        return "story collection"
    return "novel"


def main():
    all_rows = load_manifest(CHUNKED / "chunk_manifest.csv", filter_reference=True)
    ref_ids = {r["chunk_id"] for r in load_manifest(SAMPLED / "chunk_manifest.csv")}
    leftover = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        if r["chunk_id"] not in ref_ids:
            leftover[r["author"]][r["work"]].append(r)

    rng = random.Random(SEED)
    anchors, spares = [], []
    for author in sorted(leftover):
        books = leftover[author]
        picked = stratified_sample(books, ANCHORS_PER_AUTHOR, rng)
        assert len(picked) == ANCHORS_PER_AUTHOR, (author, len(picked))
        picked_ids = {r["chunk_id"] for r in picked}
        (OUT / author).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(sorted(picked, key=lambda r: (r["work"], int(r["chunk_number"]))), 1):
            shutil.copy(CHUNKED / author / f"{r['chunk_id']}.txt", OUT / author / f"{r['chunk_id']}.txt")
            anchors.append({"anchor_id": f"{author}_A{i:02d}", "chunk_id": r["chunk_id"],
                            "author": author, "work": r["work"], "form": form_of(author, r["work"]),
                            "chunk_number": r["chunk_number"], "word_count": r["word_count"]})
        spares += [r for w in books.values() for r in w if r["chunk_id"] not in picked_ids]

    write_manifest(OUT / "anchor_manifest.csv", anchors)
    write_manifest(OUT / "spare_manifest.csv", spares)

    by_author = defaultdict(list)
    for a in anchors:
        by_author[a["author"]].append(a["anchor_id"])
    scen = list(csv.DictReader(open("scenarios_final.csv", encoding="utf-8")))
    paired = []
    for author in sorted(by_author):
        rows = sorted((r for r in scen if r["author"] == author), key=lambda r: r["scenario_id"])
        slots = by_author[author] * USES_PER_ANCHOR
        assert len(slots) == len(rows), (author, len(slots), len(rows))
        rng.shuffle(slots)
        for r, a in zip(rows, slots):
            paired.append({**r, "anchor_id": a})
    with open("scenarios_final_paired.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader()
        w.writerows(paired)

    print(f"{len(anchors)} anchors, {len(spares)} spare chunks, {len(paired)} paired scenarios")


if __name__ == "__main__":
    main()
