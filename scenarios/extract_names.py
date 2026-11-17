"""Build per-author banned-name lists from the standardised books.

Runs spaCy NER over every book and keeps recurring proper names
(PERSON, GPE, LOC, FAC, ORG) per author. Output feeds the source-overlap
check in check_scenarios.py.

Usage: .venv/Scripts/python extract_names.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import spacy

BOOKS_DIR = Path("books/1_standardized_books")
OUT = Path("banned_names.json")
LABELS = {"PERSON", "GPE", "LOC", "FAC", "ORG"}
MIN_COUNT = 3          # name must recur in an author's works to be banned
CHUNK_CHARS = 100_000  # spaCy max_length safety

nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


def author_of(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def chunks(text: str):
    for i in range(0, len(text), CHUNK_CHARS):
        yield text[i:i + CHUNK_CHARS]


def clean(name: str) -> str:
    name = re.sub(r"^(the|a|an)\s+", "", name.strip(), flags=re.I)
    name = re.sub(r"'s$", "", name)
    return re.sub(r"\s+", " ", name).strip(" .,;:!?\"'")


counts: dict[str, Counter] = defaultdict(Counter)
works: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

for path in sorted(BOOKS_DIR.glob("*.txt")):
    author = author_of(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    for doc in nlp.pipe(chunks(text), batch_size=4):
        for ent in doc.ents:
            if ent.label_ not in LABELS:
                continue
            name = clean(ent.text)
            if len(name) < 3 or not name[0].isupper():
                continue
            counts[author][name] += 1
            works[author][name].add(path.stem)
    print(f"done {path.name}", flush=True)

result = {
    author: sorted(
        ({"name": n, "count": c, "works": sorted(works[author][n])}
         for n, c in cnt.items() if c >= MIN_COUNT),
        key=lambda d: -d["count"],
    )
    for author, cnt in sorted(counts.items())
}
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
print({a: len(v) for a, v in result.items()})
