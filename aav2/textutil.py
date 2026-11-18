"""Text normalisation, paragraph detection, tokenisation, chunking and truncation.

Ported unchanged in behaviour from aa_pipeline.stylolib (text.py, chunking.py),
so reference books and generated texts are processed with identical geometric rules.
"""
from __future__ import annotations

import re


def normalise_text(text: str) -> str:
    """Normalise line endings and collapse excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_paragraphs(text: str, min_density: int = 2) -> list[str]:
    """Double-newline split; fall back to single newline if paragraph density is too low."""
    expected_min = max(1, (len(text.split()) / 10_000) * min_density)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= expected_min:
        return paras
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) >= expected_min:
        return paras
    return [text] if text.strip() else []


def tokenise(text: str, pattern: str) -> list[str]:
    """Extract lowercase alphabetic word tokens."""
    return re.findall(pattern, text.lower())


def _cumulative(paragraphs: list[str]) -> list[int]:
    """Return cumulative word counts per paragraph (inclusive)."""
    out, running = [], 0
    for p in paragraphs:
        running += len(p.split())
        out.append(running)
    return out


def _sentence_or_word_cut(words: list[str], target: int, window: int, sent_re: re.Pattern) -> int:
    """Find a sentence cut within window, or fallback to word cut."""
    lo, hi = max(0, target - window), min(len(words), target + window)
    best, best_dist = None, float("inf")
    for i in range(lo + 1, hi + 1):
        if sent_re.search(words[i - 1]) and abs(i - target) < best_dist:
            best, best_dist = i, abs(i - target)
    return best if best is not None else min(target, len(words))


def _sentence_or_word_cut_with_boundary(
    words: list[str], target: int, window: int, sent_re: re.Pattern
) -> tuple[int, str]:
    """Find a sentence or word cut near target, returning (cut_index, boundary_type)."""
    lo = max(0, target - window)
    hi = min(len(words), target + window)
    best_sent = None
    best_dist = float("inf")

    for i in range(lo + 1, hi + 1):
        if sent_re.search(words[i - 1]):
            dist = abs(i - target)
            if dist < best_dist:
                best_dist = dist
                best_sent = i

    if best_sent is not None:
        return best_sent, "sentence"

    return min(target, len(words)), "word"


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


def chunk_paragraphs(
    paragraphs: list[str],
    target: int = 1000,
    window: int = 0,
    min_size: int = 1000,
    sentence_end_re: str | re.Pattern = r"[.!?]$",
) -> list[tuple[str, str]]:
    """Slice paragraph list into chunks of ~target words, preserving paragraph structure.

    Returns a list of (chunk_text, boundary_type) tuples.
    The final fragment is retained only if >= min_size words.
    """
    paras = list(paragraphs)
    cumulative = _cumulative(paras)
    sent_re = re.compile(sentence_end_re) if isinstance(sentence_end_re, str) else sentence_end_re

    chunks: list[tuple[str, str]] = []
    para_start = 0

    while para_start < len(paras):
        word_start = cumulative[para_start - 1] if para_start > 0 else 0
        total_words = cumulative[-1]
        remaining_words = total_words - word_start
        word_target = word_start + target

        # ── Final fragment ────────────────────────────────────────────────
        if remaining_words < target + window:
            if remaining_words >= min_size:
                chunk_text = "\n\n".join(paras[para_start:])
                chunks.append((chunk_text, "paragraph"))
            break

        # ── Search for nearest paragraph boundary in the window ──────────
        best_para_idx = None
        best_para_dist = float("inf")

        for i in range(para_start, len(paras)):
            w = cumulative[i]
            dist = abs(w - word_target)

            if w < word_target - window:
                continue
            if w > word_target + window:
                break

            if dist < best_para_dist:
                best_para_dist = dist
                best_para_idx = i

        if best_para_idx is not None:
            para_end = best_para_idx + 1
            chunk_text = "\n\n".join(paras[para_start:para_end])
            chunks.append((chunk_text, "paragraph"))
            para_start = para_end
            continue

        # ── Fallback: paragraph straddles the window; split it ───────────
        split_idx = para_start
        for i in range(para_start, len(paras)):
            if cumulative[i] >= word_target - window:
                split_idx = i
                break

        words = paras[split_idx].split()
        para_word_start = cumulative[split_idx] - len(words)
        local_target = word_target - para_word_start

        cut, btype = _sentence_or_word_cut_with_boundary(
            words, local_target, window=window, sent_re=sent_re
        )

        head = " ".join(words[:cut])
        tail = " ".join(words[cut:])

        before_paras = paras[para_start:split_idx]
        if head:
            before_paras = before_paras + [head]
        chunk_text = "\n\n".join(before_paras)
        chunks.append((chunk_text, btype))

        new_paras = ([tail] if tail else []) + paras[split_idx + 1:]
        paras = paras[:split_idx] + new_paras
        cumulative = _cumulative(paras)
        para_start = split_idx

    return chunks
