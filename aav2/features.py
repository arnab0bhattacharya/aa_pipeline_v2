"""Stage `extract`: parse every document once and compute its fold-independent features.

Refactored from the original feature_extractors package. Changes, as agreed:
  * one pinned spaCy model (en_core_web_sm 3.8.0 on spaCy 3.8.7) for everything,
    including the POS n-gram vocabulary (previously chosen with sm, counted with lg)
  * semantic_coherence (GloVe, en_core_web_lg) dropped
  * max_parse_depth dropped (single-sentence outlier; redundant with avg_max_depth)
  * passive voice counted per clause (a clause is passive if its head has an
    auxpass/nsubjpass/csubjpass child), not per dependency label
  * pyphen required; no heuristic syllable fallback
  * no exception swallowing: NaN is returned only for defined edge cases
  * punctuation kept as before in sentence length and POS ratios

Fold-dependent features (MFW and n-gram relative frequencies) are not computed
here; each document's alpha tokens and POS tags are cached so every reference
set (full, LOO, LOWO) can rebuild its own vocabulary (see space.py).

Outputs (out/):
  fixed_features.csv   doc_id + 29 features
  sequences.pkl        {doc_id: (alpha_tokens, pos_tags_without_punct)}
"""
from __future__ import annotations

import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sstats
from tqdm import tqdm

from .config import Config
from .textutil import tokenise

SPACY_VERSION = "3.8.7"
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_VERSION = "3.8.0"
PYPHEN_LANG = "en_GB"

FIXED_FEATURES = [
    # lexical
    "herdans_c", "simpsons_index", "hapax_ratio", "avg_word_length", "word_length_std",
    # syntactic
    "avg_sentence_length", "sentence_length_std", "avg_parse_depth", "passive_voice_ratio",
    "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio", "pos_adv_ratio", "pos_pronoun_ratio",
    # structural
    "num_paragraphs", "avg_paragraph_length",
    # readability
    "avg_syllables_per_word", "complex_word_ratio",
    # entropy
    "word_entropy", "normalized_word_entropy", "sentence_length_entropy", "bigram_entropy",
    # moments
    "wl_skew", "wl_kurt", "sl_skew", "sl_kurt",
    # information density
    "info_density_gradient", "density_trend",
    # dependency tree
    "avg_max_depth",
]

_SUBJ = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
_PASSIVE = {"auxpass", "nsubjpass", "csubjpass"}
NAN = float("nan")


def load_nlp():
    import spacy
    if spacy.__version__ != SPACY_VERSION:
        raise RuntimeError(f"spaCy {spacy.__version__} found; pipeline is pinned to {SPACY_VERSION}")
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])
    if nlp.meta["version"] != SPACY_MODEL_VERSION:
        raise RuntimeError(f"{SPACY_MODEL} {nlp.meta['version']} found; pinned to {SPACY_MODEL_VERSION}")
    return nlp


def load_syllable_counter():
    import pyphen  # required: ImportError is the intended failure
    dic = pyphen.Pyphen(lang=PYPHEN_LANG)

    def count(word: str) -> int:
        return len(dic.positions(word)) + 1

    return count


def _shannon(counts: Counter, total: int) -> float:
    if total == 0:
        return NAN
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n > 0)


def _pop_std(values: list[float]) -> float:
    return float(np.std(values)) if values else NAN


def _skew_kurt(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return NAN, NAN
    if np.std(values) == 0.0:
        return 0.0, -3.0
    return float(sstats.skew(values, bias=True)), float(sstats.kurtosis(values, bias=True))


def _depth(token) -> int:
    depth, cur = 0, token
    while cur.head.i != cur.i:
        cur = cur.head
        depth += 1
    return depth


def doc_features(doc, text: str, tokens: list[str], syllables) -> dict[str, float]:
    f: dict[str, float] = {}
    N = len(tokens)
    if N < 2:
        raise ValueError(f"Document has {N} alpha tokens; cannot compute features")

    # lexical
    counts = Counter(tokens)
    V = len(counts)
    wl = [len(w) for w in tokens]
    f["herdans_c"] = math.log(V) / math.log(N)
    f["simpsons_index"] = sum(n * (n - 1) for n in counts.values()) / (N * (N - 1))
    f["hapax_ratio"] = sum(1 for n in counts.values() if n == 1) / V
    f["avg_word_length"] = float(np.mean(wl))
    f["word_length_std"] = _pop_std(wl)

    # syntactic (sentence length counts every non-space token, punctuation included)
    sents = list(doc.sents)
    sl = [sum(1 for t in s if not t.is_space) for s in sents]
    f["avg_sentence_length"] = float(np.mean(sl)) if sl else NAN
    f["sentence_length_std"] = _pop_std(sl)
    depth = {t.i: _depth(t) for t in doc if not t.is_punct and not t.is_space}
    f["avg_parse_depth"] = float(np.mean(list(depth.values()))) if depth else NAN

    clause_heads = [t for t in doc if t.pos_ in ("VERB", "AUX")
                    and (t.dep_ == "ROOT" or any(c.dep_ in _SUBJ for c in t.children))]
    passive = sum(1 for t in clause_heads if any(c.dep_ in _PASSIVE for c in t.children))
    f["passive_voice_ratio"] = passive / len(clause_heads) if clause_heads else NAN

    non_space = [t for t in doc if not t.is_space]
    pos = Counter(t.pos_ for t in non_space)
    for name, tag in (("noun", "NOUN"), ("verb", "VERB"), ("adj", "ADJ"), ("adv", "ADV"), ("pronoun", "PRON")):
        f[f"pos_{name}_ratio"] = pos.get(tag, 0) / len(non_space)

    # structural
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    f["num_paragraphs"] = float(len(paras))
    f["avg_paragraph_length"] = float(np.mean([len(p.split()) for p in paras]))

    # readability
    syl = [syllables(w) for w in tokens]
    f["avg_syllables_per_word"] = float(np.mean(syl))
    f["complex_word_ratio"] = sum(1 for s in syl if s >= 3) / N

    # entropy
    f["word_entropy"] = _shannon(counts, N)
    f["normalized_word_entropy"] = f["word_entropy"] / math.log2(V) if V >= 2 else NAN
    f["sentence_length_entropy"] = _shannon(Counter(sl), len(sl))
    bigrams = Counter(zip(tokens, tokens[1:]))
    f["bigram_entropy"] = _shannon(bigrams, N - 1)

    # moments
    f["wl_skew"], f["wl_kurt"] = _skew_kurt(wl)
    f["sl_skew"], f["sl_kurt"] = _skew_kurt(sl)

    # information density across sentences
    dens = []
    for s in sents:
        words = [t.lower_ for t in s if not t.is_space and not t.is_punct]
        if words:
            dens.append(len(set(words)) / len(words))
    if len(dens) >= 2:
        f["info_density_gradient"] = float(np.mean(np.abs(np.diff(dens))))
        j = np.arange(len(dens))
        denom = float(((j - j.mean()) ** 2).sum())
        f["density_trend"] = float(((j - j.mean()) * (np.array(dens) - np.mean(dens))).sum() / denom)
    else:
        f["info_density_gradient"] = f["density_trend"] = NAN

    # dependency tree: mean over sentences of the deepest token
    maxd = [max(depth[t.i] for t in s if t.i in depth) for s in sents if any(t.i in depth for t in s)]
    f["avg_max_depth"] = float(np.mean(maxd)) if maxd else NAN

    return f


def run(cfg: Config) -> None:
    docs = pd.read_csv(cfg.out / "documents.csv")
    nlp = load_nlp()
    syllables = load_syllable_counter()

    texts = [open(p, encoding="utf-8", errors="strict").read() for p in docs["path"]]
    rows, sequences = [], {}
    for doc_id, text, doc in tqdm(zip(docs["doc_id"], texts, nlp.pipe(texts, batch_size=32)),
                                  total=len(texts), desc="extract", unit="doc"):
        tokens = tokenise(text, cfg.token_regex)
        feats = doc_features(doc, text, tokens, syllables)
        rows.append({"doc_id": doc_id, **feats})
        sequences[doc_id] = (tokens, [t.pos_ for t in doc if not t.is_punct and not t.is_space])

    fixed = pd.DataFrame(rows)[["doc_id"] + FIXED_FEATURES]
    if len(fixed) != len(docs):
        raise ValueError("Feature rows do not match documents")
    fixed.to_csv(cfg.out / "fixed_features.csv", index=False)
    with open(cfg.out / "sequences.pkl", "wb") as f:
        pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

    nan = fixed[FIXED_FEATURES].isna().sum()
    print(f"extract: {len(fixed)} docs x {len(FIXED_FEATURES)} fixed features; "
          f"NaN cells: {dict(nan[nan > 0]) or 0}")
