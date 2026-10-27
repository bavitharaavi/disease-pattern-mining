"""Associative classification for multi-disease detection.

AssociativeClassifier is a CBA-style multi-label classifier: for each
disease it keeps the mined rules that predict that disease and flags a
patient when any sufficiently confident rule fires. It is fully
interpretable — every prediction can be explained by the rule that fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .apriori import apriori
from .fpgrowth import fpgrowth
from .rules import Rule, generate_rules, prune_redundant


@dataclass
class AssociativeClassifier:
    diseases: Sequence[str]
    min_support: float = 0.05
    min_confidence: float = 0.6
    min_lift: float = 1.2
    max_len: int = 4
    algorithm: str = "fpgrowth"
    rules_: Dict[str, List[Rule]] = field(default_factory=dict, init=False)

    def _labels(self) -> List[str]:
        return [f"DX:{d}" for d in self.diseases]

    def fit(self, transactions: List[List[str]]) -> "AssociativeClassifier":
        miner = fpgrowth if self.algorithm == "fpgrowth" else apriori
        frequent = miner(transactions, self.min_support, self.max_len)
        all_rules = generate_rules(frequent, self.min_confidence, self.min_lift,
                                   consequent_filter=self._labels())
        all_rules = prune_redundant(all_rules)
        self.rules_ = {d: [r for r in all_rules if r.consequent == frozenset([f"DX:{d}"])]
                       for d in self.diseases}
        return self

    @staticmethod
    def _strip_labels(t: List[str]) -> frozenset:
        return frozenset(i for i in t if not i.startswith("DX:"))

    def predict_proba(self, transactions: List[List[str]]) -> np.ndarray:
        """Score = confidence of the strongest matching rule (0 if none fires)."""
        out = np.zeros((len(transactions), len(self.diseases)))
        for i, t in enumerate(transactions):
            items = self._strip_labels(t)
            for j, d in enumerate(self.diseases):
                confs = [r.confidence for r in self.rules_.get(d, []) if r.antecedent <= items]
                out[i, j] = max(confs) if confs else 0.0
        return out

    def predict(self, transactions: List[List[str]], threshold: Optional[float] = None) -> np.ndarray:
        thr = self.min_confidence if threshold is None else threshold
        return (self.predict_proba(transactions) >= thr).astype(int)

    def explain(self, transaction: List[str]) -> Dict[str, Optional[Rule]]:
        items = self._strip_labels(transaction)
        result = {}
        for d in self.diseases:
            fired = [r for r in self.rules_.get(d, []) if r.antecedent <= items]
            result[d] = max(fired, key=lambda r: r.confidence) if fired else None
        return result
