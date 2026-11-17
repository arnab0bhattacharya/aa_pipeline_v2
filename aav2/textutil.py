"""Text normalisation, paragraph detection, tokenisation and first-chunk truncation.

Ported unchanged in behaviour from aa_pipeline.stylolib (text.py, chunking.py),
so generated texts are cut exactly the way the reference chunks were.
"""
from __future__ import annotations

import re


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_paragraphs(text: str, min_density: int = 2) -> list[str]:
    """Double-newline split; fall back to single newline if too few paragraphs."""
    expected_min = max(1, (len(text.split()) / 10_000) * min_density)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= expected_min:
        return paras
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) >= expected_min:
        return paras
    return [text] if text.strip() else []


def tokenise(text: str, pattern: str) -> list[str]:
    return re.findall(pattern, text.lower())


def _cumulative(paragraphs: list[str]) -> list[int]:
    out, running = [], 0
    for p in paragraphs:
        running += len(p.split())
        out.append(running)
    return out


def _sentence_or_word_cut(words: list[str], target: int, window: int, sent_re: re.Pattern) -> int:
    lo, hi = max(0, target - window), min(len(words), target + window)
    best, best_dist = None, float("inf")
    for i in range(lo + 1, hi + 1):
        if sent_re.search(words[i - 1]) and abs(i - target) < best_dist:
            best, best_dist = i, abs(i - target)
    return best if best is not None else min(target, len(words))


def extract_first_chunk(paragraphs: list[str], target: int, window: int, sentence_end_re: str) -> str:
    """First ~target words, cut at the nearest paragraph, else sentence, else word boundary."""
    cumulative = _cumulative(paragraphs)
    total = cumulative[-1] if cumulative else 0
    if total <= target + window:
        return "\n\n".join(paragraphs)

    best, best_dist = None, float("inf")
    for i, w in enumerate(cumulative):
        if w < target - window:
            continue
        if w > target + window:
            break
        if abs(w - target) < best_dist:
            best, best_dist = i, abs(w - target)
    if best is not None:
        return "\n\n".join(paragraphs[: best + 1])

    split_idx = next(i for i, w in enumerate(cumulative) if w >= target - window)
    words = paragraphs[split_idx].split()
    local_target = target - (cumulative[split_idx] - len(words))
    cut = _sentence_or_word_cut(words, local_target, window, re.compile(sentence_end_re))
    head = " ".join(words[:cut])
    return "\n\n".join(paragraphs[:split_idx] + ([head] if head else []))
