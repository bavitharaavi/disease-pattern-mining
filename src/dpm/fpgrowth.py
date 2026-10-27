"""FP-Growth frequent-itemset mining.

FP-Growth (Han, Pei & Yin, 2000) avoids Apriori's repeated candidate
generation by compressing the dataset into a prefix tree (FP-tree) and
mining conditional trees recursively. It is typically much faster on
dense medical data with many co-occurring attributes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

Itemset = FrozenSet[str]


class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: Optional[str], parent: Optional["_Node"]):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: Dict[str, "_Node"] = {}
        self.link: Optional["_Node"] = None


class FPTree:
    def __init__(self, weighted_transactions: List[Tuple[List[str], int]], min_count: float):
        counts: Dict[str, int] = defaultdict(int)
        for items, w in weighted_transactions:
            for it in items:
                counts[it] += w
        self.frequent = {i: c for i, c in counts.items() if c >= min_count}
        self.root = _Node(None, None)
        self.header: Dict[str, _Node] = {}
        self._tails: Dict[str, _Node] = {}

        # Deterministic order: descending count, then name.
        order = sorted(self.frequent, key=lambda i: (-self.frequent[i], i))
        self.rank = {item: r for r, item in enumerate(order)}

        for items, w in weighted_transactions:
            filtered = sorted((i for i in set(items) if i in self.frequent), key=self.rank.__getitem__)
            if filtered:
                self._insert(filtered, w)

    def _insert(self, items: List[str], weight: int) -> None:
        node = self.root
        for it in items:
            child = node.children.get(it)
            if child is None:
                child = _Node(it, node)
                node.children[it] = child
                if it in self._tails:
                    self._tails[it].link = child
                else:
                    self.header[it] = child
                self._tails[it] = child
            child.count += weight
            node = child

    def prefix_paths(self, item: str) -> List[Tuple[List[str], int]]:
        paths = []
        node = self.header.get(item)
        while node is not None:
            path, parent = [], node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                paths.append((path[::-1], node.count))
            node = node.link
        return paths


def _mine(tree: FPTree, suffix: Itemset, min_count: float, out: Dict[Itemset, int], max_len):
    # Process least-frequent items first (standard bottom-up order).
    for item in sorted(tree.frequent, key=lambda i: (tree.frequent[i], i)):
        new_set = suffix | {item}
        out[new_set] = tree.frequent[item]
        if max_len is not None and len(new_set) >= max_len:
            continue
        cond = FPTree(tree.prefix_paths(item), min_count)
        if cond.frequent:
            _mine(cond, new_set, min_count, out, max_len)


def fpgrowth(
    transactions: Iterable[Iterable[str]],
    min_support: float = 0.1,
    max_len: int | None = None,
) -> Dict[Itemset, float]:
    """Return every frequent itemset mapped to its support (0..1)."""
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    tx = [list(t) for t in transactions]
    n_tx = len(tx)
    if n_tx == 0:
        return {}
    min_count = min_support * n_tx
    tree = FPTree([(t, 1) for t in tx], min_count)
    counts: Dict[Itemset, int] = {}
    _mine(tree, frozenset(), min_count, counts, max_len)
    return {s: c / n_tx for s, c in counts.items()}
