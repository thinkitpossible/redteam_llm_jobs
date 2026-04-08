from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[，。、“”‘’；：！？,.!?:;()\[\]{}<>《》/\\|_\-]+")


def normalize_profession_name(name: str) -> str:
    normalized = name.replace("\u3000", " ").strip()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized


def normalize_for_similarity(text: str, profession_name: str | None = None) -> str:
    normalized = text.strip()
    if profession_name:
        normalized = normalized.replace(profession_name, "<职业>")
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.lower().strip()


def stable_id(prefix: str, *parts: object, length: int = 10) -> str:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def char_ngram_counter(text: str, n: int = 3) -> Counter[str]:
    if len(text) < n:
        return Counter({text: 1}) if text else Counter()
    return Counter(text[index : index + n] for index in range(len(text) - n + 1))


def cosine_similarity(left: str, right: str, n: int = 3) -> float:
    left_counter = char_ngram_counter(left, n=n)
    right_counter = char_ngram_counter(right, n=n)
    if not left_counter or not right_counter:
        return 0.0
    common = set(left_counter).intersection(right_counter)
    numerator = sum(left_counter[token] * right_counter[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counter.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counter.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
