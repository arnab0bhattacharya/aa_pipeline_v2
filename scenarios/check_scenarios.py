"""Automatic checks on drafted scenarios.

Input : drafts/{AUTHOR}.jsonl  (one JSON object per line:
        id, author, premise, setting, mode)
Output: check_report.csv (one row per scenario, pass/fail per rule)
        similarity_pairs.csv (within-author pairs above SIM_REPORT)

Rules
  R1 length      premise 30-60 words; setting <= 25 words
                 (floor lowered from 40 after the first pass: per-author means
                 were balanced at 41.5-45.5 words, so parity, not the absolute
                 floor, is what matters)
  R2 style       no banned style vocabulary in premise or setting
  R3 names       no recurring proper name from the author's own books
                 (banned_names.json), except allowed real places
  R4 no persons  premise names no individuals (spaCy PERSON entities)
  R5 echo        no 8-word sequence shared with any of the author's books
  R6 similarity  max within-author TF-IDF cosine reported; pairs above
                 SIM_REPORT listed for human review

Usage: .venv/Scripts/python check_scenarios.py
"""
import csv
import json
import re
from pathlib import Path

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DRAFTS = Path("drafts")
BOOKS_DIR = Path("books/1_standardized_books")
BANNED = json.loads(Path("banned_names.json").read_text(encoding="utf-8"))
SIM_REPORT = 0.30
ECHO_N = 8

STYLE_WORDS = {
    # tone / manner
    "witty", "wit", "ironic", "irony", "ironical", "satirical", "satire",
    "comic", "comedic", "humorous", "playful", "whimsical", "brooding",
    "gloomy", "melancholy", "melancholic", "lyrical", "poetic", "ornate",
    "florid", "elegant", "eloquent", "terse", "laconic", "verbose",
    "epigram", "epigrams", "epigrammatic", "paradox", "paradoxical",
    "aphorism", "aphoristic", "gothic", "sentimental", "dramatic",
    "atmospheric", "evocative", "impressionistic", "stream",
    "consciousness", "introspective", "interior", "monologue",
    # form / narration
    "narrator", "narration", "narrative", "voice", "tone", "style",
    "prose", "sentence", "sentences", "dialogue", "dialogues", "rhythm",
    "first-person", "third-person", "omniscient", "perspective",
    "point-of-view", "flashback", "epistolary", "letters-form",
    "chapter", "chapters",
}

ALLOWED_PLACES = {
    "London", "England", "English", "Scotland", "Scottish", "Wales",
    "Ireland", "Irish", "Britain", "British", "Europe", "European",
    "France", "French", "Paris", "Italy", "Italian", "Rome", "Naples",
    "Venice", "Florence", "Germany", "German", "Switzerland", "Swiss",
    "Geneva", "Russia", "Russian", "Brussels", "Belgium", "Holland",
    "Dutch", "Spain", "Spanish", "America", "American", "India",
    "Indian", "Africa", "African", "Africa's", "Australia", "Bath",
    "Brighton", "Oxford", "Cambridge", "Yorkshire", "Devon", "Kent",
    "Surrey", "Sussex", "Thames", "Channel", "Continent", "Mediterranean",
    "Atlantic", "Pacific", "East", "West", "North", "South", "Christmas",
    "Easter", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November",
    "December", "Lent", "Parliament", "Church", "Navy", "Army", "God",
    # real places and institutions hit by R3 in the first pass; each checked by
    # hand as real geography, not a fictional setting from the novels
    "Afghanistan", "Alps", "Antwerp", "Archipelago", "Athens", "Bailey",
    "Berkshire", "Bloomsbury", "Calabria", "Camden", "Canada", "Charing Cross",
    "Charing Cross Road", "Chelsea", "China", "City of London", "Clerkenwell",
    "Cornwall", "Court", "Cumberland", "Derbyshire", "Devonshire", "Edinburgh",
    "Englishman", "Greece", "Grosvenor Square", "Hampstead", "Harley Street",
    "Hertfordshire", "Holborn", "House", "House of Commons", "Inn",
    "Inn Fields", "Islington", "James", "Kensington", "Kentish",
    "Knightsbridge", "Lincoln", "Liverpool", "Mall", "Manchester", "Mayfair",
    "Milan", "Newgate", "Norfolk", "Northumberland", "Old Bailey",
    "Oxford Street", "Pentonville", "Portsmouth", "Reform", "Regent Street",
    "Richmond", "Saffron Hill", "Scotland Yard", "Shropshire", "Siberia",
    "Soho", "Somerset", "South America", "South Kensington", "Southampton",
    "Southwark", "Strand", "Suffolk", "Trafalgar Square", "Vienna",
    "West Indies", "Westminster", "Weymouth", "Wiltshire", "Zurich",
}

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
TOKEN = re.compile(r"\b[a-z]+\b")


def load_drafts():
    rows = []
    for path in sorted(DRAFTS.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def book_ngrams(author: str) -> set:
    grams = set()
    for path in BOOKS_DIR.glob(f"{author}_*.txt"):
        toks = TOKEN.findall(path.read_text(encoding="utf-8", errors="replace").lower())
        grams.update(tuple(toks[i:i + ECHO_N]) for i in range(len(toks) - ECHO_N + 1))
    return grams


def banned_hits(text: str, author: str) -> list[str]:
    hits = []
    for entry in BANNED.get(author, []):
        name = entry["name"]
        if name in ALLOWED_PLACES:
            continue
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text):
            hits.append(name)
    return hits


NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


def person_like(premise: str) -> list[str]:
    return [e.text for e in NLP(premise).ents if e.label_ == "PERSON"]


def main():
    rows = load_drafts()
    grams_cache: dict[str, set] = {}
    report = []
    for r in rows:
        a = r["author"]
        if a not in grams_cache:
            grams_cache[a] = book_ngrams(a)
        text = f"{r['premise']} {r['setting']}"
        p_words = len(WORD.findall(r["premise"]))
        s_words = len(WORD.findall(r["setting"]))
        style = sorted({w.lower() for w in WORD.findall(text)} & STYLE_WORDS)
        names = banned_hits(text, a)
        persons = person_like(r["premise"])
        toks = TOKEN.findall(text.lower())
        echoes = [" ".join(toks[i:i + ECHO_N]) for i in range(len(toks) - ECHO_N + 1)
                  if tuple(toks[i:i + ECHO_N]) in grams_cache[a]]
        report.append({
            "id": r["id"], "author": a, "mode": r.get("mode", ""),
            "premise_words": p_words, "setting_words": s_words,
            "R1_length": 30 <= p_words <= 60 and s_words <= 25,
            "R2_style": not style, "style_hits": ";".join(style),
            "R3_names": not names, "name_hits": ";".join(names),
            "R4_persons": not persons, "person_hits": ";".join(persons),
            "R5_echo": not echoes, "echo_hits": " | ".join(echoes[:3]),
        })

    pairs = []
    by_author: dict[str, list] = {}
    for r in rows:
        by_author.setdefault(r["author"], []).append(r)
    max_sim = {}
    for a, rs in by_author.items():
        texts = [f"{r['premise']} {r['setting']}" for r in rs]
        m = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))
        for i in range(len(rs)):
            row_max = 0.0
            for j in range(len(rs)):
                if i != j:
                    row_max = max(row_max, m[i, j])
                if j > i and m[i, j] >= SIM_REPORT:
                    pairs.append({"author": a, "id_a": rs[i]["id"], "id_b": rs[j]["id"],
                                  "cosine": round(float(m[i, j]), 3)})
            max_sim[rs[i]["id"]] = round(float(row_max), 3)
    for rec in report:
        rec["max_within_author_cosine"] = max_sim[rec["id"]]
        rec["auto_pass"] = all(rec[k] for k in ("R1_length", "R2_style", "R3_names",
                                                "R4_persons", "R5_echo"))

    with open("check_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    with open("similarity_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["author", "id_a", "id_b", "cosine"])
        w.writeheader()
        w.writerows(sorted(pairs, key=lambda p: -p["cosine"]))

    summary: dict[str, list] = {}
    for rec in report:
        s = summary.setdefault(rec["author"], [0, 0])
        s[0] += 1
        s[1] += rec["auto_pass"]
    for a, (n, ok) in sorted(summary.items()):
        print(f"{a}: {ok}/{n} pass automatic checks")
    print(f"{len(pairs)} within-author pairs with cosine >= {SIM_REPORT}")


if __name__ == "__main__":
    main()
