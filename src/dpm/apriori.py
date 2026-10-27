"""Apriori frequent-itemset mining.

A clean, dependency-free implementation of the classic Apriori algorithm
(Agrawal & Srikant, 1994) with the standard candidate-pruning step: a
k-itemset can only be frequent if every (k-1)-subset is frequent.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Sequence

Itemset = FrozenSet[str]


def _generate_candidates(prev_frequent: Sequence[Itemset], k: int) -> List[Itemset]:
    """Join step + prune step."""
    prev_set = set(prev_frequent)
    sorted_prev = [tuple(sorted(s)) for s in prev_frequent]
    sorted_prev.sort()
    candidates: List[Itemset] = []
    n = len(sorted_prev)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted_prev[i], sorted_prev[j]
            # Join only itemsets that share their first k-2 items.
            if a[: k - 2] != b[: k - 2]:
                break
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != k:
                continue
            # Prune: every (k-1)-subset must already be frequent.
            if all(frozenset(sub) in prev_set for sub in combinations(candidate, k - 1)):
                candidates.append(candidate)
    return candidates


def apriori(
    transactions: Iterable[Iterable[str]],
    min_support: float = 0.1,
    max_len: int | None = None,
) -> Dict[Itemset, float]:
    """Return every frequent itemset mapped to its support (0..1).

    Args:
        transactions: iterable of transactions, each an iterable of item labels.
        min_support: minimum fraction of transactions an itemset must appear in.
        max_len: optional cap on itemset size.
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")

    tx = [frozenset(t) for t in transactions]
    n_tx = len(tx)
    if n_tx == 0:
        return {}
    min_count = min_support * n_tx

    item_counts = Counter(item for t in tx for item in t)
    current = {frozenset([i]): c for i, c in item_counts.items() if c >= min_count}
    result: Dict[Itemset, float] = {s: c / n_tx for s, c in current.items()}

    k = 2
    while current and (max_len is None or k <= max_len):
        candidates = _generate_candidates(list(current), k)
        if not candidates:
            break
        counts = Counter()
        for t in tx:
            if len(t) < k:
                continue
            for c in candidates:
                if c <= t:
                    counts[c] += 1
        current = {s: c for s, c in counts.items() if c >= min_count}
        result.update({s: c / n_tx for s, c in current.items()})
        k += 1
    return result
