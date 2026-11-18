"""Select the final 50 scenarios per author from the 60 drafts.

Rule (fixed before running):
  1. Drop every row with a drafter note in review_sheet.csv (non-corpus echoes
     and borderline resemblances).
  2. Allocate 50 slots across each author's modes in proportion to the reference
     chunks behind each mode (MODE_WEIGHTS), Hamilton rounding, capped at the
     rows available; any shortfall is redistributed the same way.
  3. Within each mode, keep a seeded random subset of the required size.

Output: scenarios_final.csv (500 rows), selection_log.csv (every draft, kept or dropped and why)

Usage: .venv/Scripts/python select_scenarios.py
"""
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
PER_AUTHOR = 50

# Chunk weight behind each drafted mode (counts from the per-work chunk tallies).
# Where a work feeds two modes its count is split evenly.
MODE_WEIGHTS = {
    "ACD": {"urban-case": 104 + 112, "country-case": 59, "past-abroad": 43 + 43},
    "CB": {"foreign-school": 87 + 192, "industrial": 213, "governess": 92.5, "dependent-woman": 92.5},
    "CD": {"growing-up": 353 + 184, "historical": 135, "industrial": 102,
           "london-poor": 104, "legal-institution": 52},
    "GE": {"provincial-town": 315, "society-vocation": 308, "rural-village": 214 + 71, "family-duty": 206},
    "HGW": {"realist-social": 96, "catastrophe": 59, "experiment-community": 48,
            "remote-science": 43, "future": 32},
    "JA": {"courtship": 1, "money": 1, "misjudgment": 1, "family-embarrassment": 1, "town-visit": 1},
    "JC": {"latin-republic": 167, "sea-honour": 129, "russian-exile": 111,
           "london-conspiracy": 90, "colonial-interior": 38},
    "MS": {"future-catastrophe": 173, "guilt-secret": 150, "ambition-science": 74,
           "grief-isolation": 44, "italian-romance": 12},
    "OW": {"society": 78, "comic-tale": 39, "fairy-tale": 33 + 16},
    "VW": {"london-family": 166, "voyage-abroad": 136, "single-day": 63,
           "life-fragments": 54, "sketch": 18},
}


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


def allocate(weights: dict, available: dict, total: int) -> dict:
    alloc = {m: 0 for m in weights}
    open_modes = dict(weights)
    remaining = total
    while remaining and open_modes:
        scale = 1000  # hamilton_allocate expects integer weights
        share = hamilton_allocate({m: round(w * scale) for m, w in open_modes.items()}, remaining)
        capped = False
        for m, s in share.items():
            room = available[m] - alloc[m]
            if s >= room:
                alloc[m] += room
                remaining -= room
                del open_modes[m]
                capped = True
        if not capped:
            for m, s in share.items():
                alloc[m] += s
            remaining = 0
    return alloc


def main():
    rows = list(csv.DictReader(open("review_sheet.csv", encoding="utf-8-sig")))
    rng = random.Random(SEED)
    final, log = [], []
    for author in sorted(MODE_WEIGHTS):
        pool = defaultdict(list)
        for r in rows:
            if r["author"] != author:
                continue
            if r["drafter_note"]:
                log.append({"id": r["id"], "author": author, "mode": r["mode"],
                            "kept": False, "reason": f"flagged: {r['drafter_note']}"})
            else:
                pool[r["mode"]].append(r)
        weights = MODE_WEIGHTS[author]
        assert set(pool) <= set(weights), (author, set(pool) - set(weights))
        available = {m: len(pool.get(m, [])) for m in weights}
        alloc = allocate(weights, available, PER_AUTHOR)
        assert sum(alloc.values()) == PER_AUTHOR, (author, alloc)
        for mode in weights:
            cands = sorted(pool.get(mode, []), key=lambda r: r["id"])
            keep = set(r["id"] for r in rng.sample(cands, alloc[mode]))
            for r in cands:
                kept = r["id"] in keep
                log.append({"id": r["id"], "author": author, "mode": mode, "kept": kept,
                            "reason": "selected" if kept else f"mode quota {alloc[mode]}/{len(cands)}"})
                if kept:
                    final.append(r)
        print(author, alloc)

    final.sort(key=lambda r: r["id"])
    with open("scenarios_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "author", "mode", "premise", "setting"])
        for r in final:
            w.writerow([r["id"], r["author"], r["mode"], r["premise"], r["setting"]])
    with open("selection_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "author", "mode", "kept", "reason"])
        w.writeheader()
        w.writerows(sorted(log, key=lambda d: d["id"]))
    print(len(final), "scenarios selected")


if __name__ == "__main__":
    main()
