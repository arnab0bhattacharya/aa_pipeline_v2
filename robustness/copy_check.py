"""Copy / provenance check (Reviewer 1; Research Design v5 section 6.4, gate G8).

Do the generated texts copy or closely paraphrase (a) their own prompt anchor passage or
(b) any of the 48 Project Gutenberg source works of the ten reference authors?

Protocol (v5 6.4):
  tokens      lowercase, regex \\b[a-z]+\\b
  candidate   exact shared span >= 13 tokens, OR
              fuzzy alignment of 30 tokens with similarity >= 0.80,
              similarity = 1 - Levenshtein(a, b) / max(|a|, |b|)   (token level)
  sources     the text's own anchor; all 48 works (all ten authors)
  secondary   shared 8-grams (LCS >= 8); 50-word / 25-step Jaccard > 0.50 against own anchor
  baseline    the 250 authentic anchor chunks, checked against the works OTHER than their own
              (what chance overlap - stock phrases, quotations - looks like in genuine prose)

Fuzzy search is seed-and-verify: 30 tokens at <= 6 edits must contain at least three exact
4-grams (pigeonhole), so candidates are diagonals with >= 2 shared 4-grams that are not stock
(<= 30 occurrences across the books); each is verified by full Levenshtein over offsets +-3.

Inputs:  generation/outputs/main/texts (2,500 untruncated outputs), aa_pipeline_v2/out/prepared_generated
         (the truncated text actually analysed), anchors/, scenarios/books/1_standardized_books
Outputs: out/results/copy_check_texts.csv          one row per text (generated + anchor baseline)
         out/results/copy_check_candidates_blind.csv   adjudication queue (no author/model/arm)
         out/results/copy_check_candidates_key.csv     candidate_id -> text, for un-blinding
         out/results/copy_check_summary.csv

Run: .venv\\Scripts\\python robustness\\copy_check.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "results"
GEN_DIR = ROOT / "generation" / "outputs" / "main" / "texts"
PREP_DIR = ROOT / "out" / "prepared_generated"
BOOK_DIR = ROOT / "scenarios" / "books" / "1_standardized_books"
ANCHOR_DIR = ROOT / "anchors"
TOKEN = re.compile(r"\b[a-z]+\b")
EXACT_K, FUZZY_LEN, FUZZY_SIM, SEED_K, SEED_MAX_FREQ, SECONDARY_K = 13, 30, 0.80, 4, 30, 8
P = np.uint64(1_000_003)

vocab: dict[str, int] = {}


def ids(text: str) -> np.ndarray:
    return np.array([vocab.setdefault(t, len(vocab)) for t in TOKEN.findall(text.lower())], dtype=np.uint64)


def kgram_hash(a: np.ndarray, k: int) -> np.ndarray:
    n = len(a) - k + 1
    if n <= 0:
        return np.empty(0, np.uint64)
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(k):
            h = h * P + a[i:i + n] + np.uint64(1)
    return h


class Index:
    """Sorted k-gram hashes of a token stream, with positions."""

    def __init__(self, tokens: np.ndarray, k: int):
        h = kgram_hash(tokens, k)
        self.order = np.argsort(h, kind="stable")
        self.h = h[self.order]

    def contains(self, q: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.h, q)
        i[i == len(self.h)] = 0
        return self.h[i] == q

    def positions(self, q) -> np.ndarray:
        lo, hi = np.searchsorted(self.h, q, "left"), np.searchsorted(self.h, q, "right")
        return self.order[lo:hi]


def spans(mask: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Merge k-gram hits at the True positions of mask into maximal token spans [s, e)."""
    out, s, e = [], None, None
    for p in np.flatnonzero(mask):
        if s is not None and p <= e:
            e = p + k
        else:
            if s is not None:
                out.append((s, e))
            s, e = p, p + k
    if s is not None:
        out.append((s, e))
    return out


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def fuzzy(g: np.ndarray, src: np.ndarray, seed_idx: Index, seed_ok, exclude=None):
    """Return verified fuzzy windows [(gen_start, src_start, sim)] with sim >= FUZZY_SIM."""
    gh = kgram_hash(g, SEED_K)
    hit = seed_idx.contains(gh) & seed_ok(gh)
    diag = defaultdict(list)
    for i in np.flatnonzero(hit):
        for j in seed_idx.positions(gh[i]):
            if exclude is not None and exclude[0] <= j < exclude[1]:
                continue
            diag[(int(j) - int(i)) // 4].append(int(i))
    found, seen = [], set()
    for d, starts in diag.items():
        starts.sort()
        for a, b in zip(starts, starts[1:]):
            if b - a > FUZZY_LEN:
                continue
            i0 = max(0, min(a, len(g) - FUZZY_LEN))
            key = (i0 // 10, d)
            if key in seen or len(g) < FUZZY_LEN:
                continue
            seen.add(key)
            gw = g[i0:i0 + FUZZY_LEN].tolist()
            best = (0.0, None)
            for off in range(-3, 4):
                j0 = i0 + d * 4 + off
                for jj in range(j0 - 3, j0 + 4):
                    if jj < 0 or jj + FUZZY_LEN > len(src):
                        continue
                    sim = 1 - lev(gw, src[jj:jj + FUZZY_LEN].tolist()) / FUZZY_LEN
                    if sim > best[0]:
                        best = (sim, jj)
            if best[0] >= FUZZY_SIM:
                found.append((i0, best[1], best[0]))
    return found


def jaccard_max(g: np.ndarray, a: np.ndarray, w=50, step=25) -> float:
    aw = [set(a[i:i + w].tolist()) for i in range(0, max(1, len(a) - w + 1), step)]
    best = 0.0
    for i in range(0, max(1, len(g) - w + 1), step):
        s = set(g[i:i + w].tolist())
        for t in aw:
            best = max(best, len(s & t) / len(s | t))
    return best


def main():
    # books, concatenated with a work index
    works, chunks = [], []
    for f in sorted(BOOK_DIR.glob("*.txt")):
        t = ids(f.read_text(encoding="utf-8"))
        works.append((f.stem, len(t)))
        chunks.append(t)
    book = np.concatenate(chunks)
    bounds = np.cumsum([0] + [n for _, n in works])
    work_of = lambda pos: works[int(np.searchsorted(bounds, pos, "right") - 1)][0]
    print(f"books: {len(works)} works, {len(book):,} tokens", flush=True)

    idx13, idx8, idx4 = Index(book, EXACT_K), Index(book, SECONDARY_K), Index(book, SEED_K)
    u4, c4 = np.unique(idx4.h, return_counts=True)
    rare4 = u4[c4 <= SEED_MAX_FREQ]
    seed_ok = lambda q: np.isin(q, rare4)

    man = pd.read_csv(ANCHOR_DIR / "anchor_manifest.csv").set_index("anchor_id")
    anchor_tok = {a: ids((ANCHOR_DIR / r.author / f"{r.chunk_id}.txt").read_text(encoding="utf-8"))
                  for a, r in man.iterrows()}
    gen_meta = pd.read_csv(DSH / "aa_pipeline_v2" / "out" / "generated.csv")
    gen_meta = gen_meta[gen_meta["method"] == "burrows"].set_index("doc_id")

    rows, cands = [], []

    def check(doc_id, kind, g, anchor_id, own_work=None, prepared=None, meta=None):
        n8 = max(1, len(g) - SECONDARY_K + 1)
        excl = None
        if own_work is not None:          # authentic baseline: ignore the chunk's own source work
            k = [w for w, _ in works].index(own_work)
            excl = (bounds[k], bounds[k + 1])
        def keep(mask, idx, k):
            if excl is None:
                return mask
            out = mask.copy()
            gh = kgram_hash(g, k)
            for i in np.flatnonzero(mask):
                pos = idx.positions(gh[i])
                out[i] = bool(np.any((pos < excl[0]) | (pos >= excl[1])))
            return out
        m13 = keep(idx13.contains(kgram_hash(g, EXACT_K)), idx13, EXACT_K)
        m8 = keep(idx8.contains(kgram_hash(g, SECONDARY_K)), idx8, SECONDARY_K)
        sp13 = spans(m13, EXACT_K)
        sp8 = spans(m8, SECONDARY_K)
        a = anchor_tok[anchor_id] if anchor_id else None
        row = {"doc_id": doc_id, "kind": kind, "tokens": len(g),
               "exact13_spans_books": len(sp13), "max_exact_span_books": max((e - s for s, e in sp13), default=0),
               "shared8_spans_books": len(sp8), "max_shared_span8_books": max((e - s for s, e in sp8), default=0),
               "frac_8grams_in_books": float(m8.sum() / n8)}
        if a is not None and own_work is None:
            ia13, ia8 = Index(a, EXACT_K), Index(a, SECONDARY_K)
            s13a = spans(ia13.contains(kgram_hash(g, EXACT_K)), EXACT_K)
            m8a = ia8.contains(kgram_hash(g, SECONDARY_K))
            row.update({"exact13_spans_anchor": len(s13a),
                        "max_exact_span_anchor": max((e - s for s, e in s13a), default=0),
                        "frac_8grams_in_anchor": float(m8a.sum() / n8),
                        "max_jaccard50_anchor": jaccard_max(g, a)})
        fz = fuzzy(g, book, idx4, seed_ok, exclude=excl)
        row["fuzzy_windows_books"] = len(fz)
        row["max_fuzzy_sim_books"] = max((s for *_, s in fz), default=np.nan)
        if meta is not None:
            row.update(meta)
        rows.append(row)
        # candidate queue
        prep = prepared.tolist() if prepared is not None else None
        def in_analysed(seg):
            if prep is None:
                return np.nan
            seg = seg.tolist(); L = len(seg)
            return any(prep[i:i + L] == seg for i in range(len(prep) - L + 1))
        inv = None
        for s, e in sp13:
            pos = int(idx13.positions(kgram_hash(g[s:s + EXACT_K], EXACT_K)[0])[0])
            cands.append({"doc_id": doc_id, "kind": kind, "type": "exact", "length": e - s, "similarity": 1.0,
                          "gen_start": s, "src_pos": pos, "source_work": work_of(pos),
                          "in_analysed_text": in_analysed(g[s:e])})
        merged = []
        for i0, j0, sim in sorted(fz):
            if merged and i0 < merged[-1][0] + FUZZY_LEN:
                if sim > merged[-1][2]:
                    merged[-1] = (i0, j0, sim)
                continue
            merged.append((i0, j0, sim))
        for i0, j0, sim in merged:
            if any(s <= i0 < e for s, e in sp13):
                continue  # already covered by an exact span
            cands.append({"doc_id": doc_id, "kind": kind, "type": "fuzzy", "length": FUZZY_LEN,
                          "similarity": round(sim, 3), "gen_start": i0, "src_pos": j0,
                          "source_work": work_of(j0), "in_analysed_text": in_analysed(g[i0:i0 + FUZZY_LEN])})

    files = sorted(GEN_DIR.glob("*.txt"))
    for n, f in enumerate(files, 1):
        doc = f.stem
        m = gen_meta.loc[doc]
        g = ids(f.read_text(encoding="utf-8"))
        prep = ids((PREP_DIR / f.name).read_text(encoding="utf-8"))
        kind = m["set"]
        check(doc, kind, g, m["anchor_id"] if kind == "imitation" else None, prepared=prep,
              meta={"author": m["author"] if kind == "imitation" else m["scenario_author"],
                    "model_slot": m["model_slot"], "anchor_id": m["anchor_id"]})
        if n % 250 == 0:
            print(f"  generated {n}/{len(files)}", flush=True)
    for a, r in man.iterrows():
        check(a, "anchor_baseline", anchor_tok[a], None, own_work=f"{r.author}_{r.work}",
              meta={"author": r.author, "anchor_id": a})
    print("  anchors done", flush=True)

    # resolve source snippets for the blind queue
    inv_vocab = {v: k for k, v in vocab.items()}
    gen_tok = {}
    def toks(doc, kind):
        if doc not in gen_tok:
            if kind == "anchor_baseline":
                gen_tok[doc] = anchor_tok[doc]
            else:
                gen_tok[doc] = ids((GEN_DIR / f"{doc}.txt").read_text(encoding="utf-8"))
        return gen_tok[doc]
    words = lambda arr: " ".join(inv_vocab[int(x)] for x in arr)
    for c in cands:
        g = toks(c["doc_id"], c["kind"])
        c["generated_span"] = words(g[c["gen_start"]:c["gen_start"] + c["length"]])
        c["source_span"] = words(book[c["src_pos"]:c["src_pos"] + c["length"]])

    texts = pd.DataFrame(rows)
    texts.to_csv(OUT / "copy_check_texts.csv", index=False)
    cand = pd.DataFrame(cands)
    if len(cand):
        cand = cand.sample(frac=1, random_state=42).reset_index(drop=True)
        cand.insert(0, "candidate_id", [f"K{i:04d}" for i in range(1, len(cand) + 1)])
        cand[["candidate_id", "type", "length", "similarity", "generated_span", "source_span"]].assign(
            label="").to_csv(OUT / "copy_check_candidates_blind.csv", index=False)
        cand.drop(columns=["generated_span", "source_span"]).to_csv(OUT / "copy_check_candidates_key.csv", index=False)

    # summary per group
    def summarise(df, name):
        r = {"group": name, "n_texts": len(df),
             "texts_with_exact13_books": int((df.exact13_spans_books > 0).sum()),
             "max_exact_span_books": int(df.max_exact_span_books.max()),
             "texts_with_fuzzy_books": int((df.fuzzy_windows_books > 0).sum()),
             "texts_with_shared8_books": int((df.shared8_spans_books > 0).sum()),
             "mean_frac_8grams_in_books": df.frac_8grams_in_books.mean(),
             "max_frac_8grams_in_books": df.frac_8grams_in_books.max()}
        if "exact13_spans_anchor" in df and df.exact13_spans_anchor.notna().any():
            r.update({"texts_with_exact13_anchor": int((df.exact13_spans_anchor > 0).sum()),
                      "max_exact_span_anchor": int(df.max_exact_span_anchor.max()),
                      "mean_frac_8grams_in_anchor": df.frac_8grams_in_anchor.mean(),
                      "texts_jaccard50_anchor_gt_0.5": int((df.max_jaccard50_anchor > 0.5).sum()),
                      "max_jaccard50_anchor": df.max_jaccard50_anchor.max()})
        return r
    summ = [summarise(texts[texts.kind == k], k) for k in ("imitation", "control", "anchor_baseline")]
    imi = texts[texts.kind == "imitation"]
    summ += [summarise(imi[imi.model_slot == m], f"imitation_{m}") for m in sorted(imi.model_slot.unique())]
    summ += [summarise(imi[imi.author == a], f"imitation_{a}") for a in sorted(imi.author.unique())]
    pd.DataFrame(summ).to_csv(OUT / "copy_check_summary.csv", index=False)
    print(pd.DataFrame(summ).to_string())
    print(f"candidates: {len(cand)}")


if __name__ == "__main__":
    main()
