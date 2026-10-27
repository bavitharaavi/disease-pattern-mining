"""Association rule generation and filtering."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional

Itemset = FrozenSet[str]


@dataclass(frozen=True)
class Rule:
    antecedent: Itemset
    consequent: Itemset
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float

    def __str__(self) -> str:
        lhs = ", ".join(sorted(self.antecedent))
        rhs = ", ".join(sorted(self.consequent))
        return (f"{{{lhs}}} => {{{rhs}}}  "
                f"(sup={self.support:.3f}, conf={self.confidence:.3f}, lift={self.lift:.2f})")

    def to_dict(self) -> dict:
        return {
            "antecedent": " & ".join(sorted(self.antecedent)),
            "consequent": " & ".join(sorted(self.consequent)),
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
            "leverage": round(self.leverage, 4),
            "conviction": round(self.conviction, 4) if self.conviction != float("inf") else "inf",
        }


def generate_rules(
    frequent: Dict[Itemset, float],
    min_confidence: float = 0.6,
    min_lift: float = 1.0,
    consequent_filter: Optional[Iterable[str]] = None,
) -> List[Rule]:
    """Generate rules A => B from frequent itemsets.

    Args:
        frequent: output of apriori()/fpgrowth().
        min_confidence: minimum P(B | A).
        min_lift: minimum lift (1.0 = independence).
        consequent_filter: if given, keep only rules whose consequent is a single
            item from this set (e.g. the disease labels) — this is how we turn
            generic pattern mining into disease-detection rules.
    """
    targets = set(consequent_filter) if consequent_filter is not None else None
    rules: List[Rule] = []
    for itemset, sup in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for cons in combinations(itemset, r):
                consequent = frozenset(cons)
                if targets is not None and (len(consequent) != 1 or not consequent <= targets):
                    continue
                antecedent = itemset - consequent
                if targets is not None and antecedent & targets:
                    continue  # don't predict a disease from another disease label
                sup_a = frequent.get(antecedent)
                sup_c = frequent.get(consequent)
                if not sup_a or not sup_c:
                    continue
                conf = sup / sup_a
                if conf < min_confidence:
                    continue
                lift = conf / sup_c
                if lift < min_lift:
                    continue
                leverage = sup - sup_a * sup_c
                conviction = float("inf") if conf >= 1 else (1 - sup_c) / (1 - conf)
                rules.append(Rule(antecedent, consequent, sup, conf, lift, leverage, conviction))
    rules.sort(key=lambda r: (-r.lift, -r.confidence, -r.support, sorted(r.antecedent)))
    return rules


def prune_redundant(rules: List[Rule]) -> List[Rule]:
    """Drop a rule if a more general rule (subset antecedent, same consequent)
    already has equal-or-higher confidence. Keeps the rule set interpretable."""
    kept: List[Rule] = []
    by_cons: Dict[Itemset, List[Rule]] = {}
    for rule in sorted(rules, key=lambda r: len(r.antecedent)):
        general = by_cons.get(rule.consequent, [])
        if any(g.antecedent < rule.antecedent and g.confidence >= rule.confidence for g in general):
            continue
        general.append(rule)
        by_cons[rule.consequent] = general
        kept.append(rule)
    kept.sort(key=lambda r: (-r.lift, -r.confidence, -r.support))
    return kept
