"""Feature selection driven by mined association rules.

Idea: a clinical feature is useful for disease detection if it appears in
many strong rules whose consequent is a disease. We score each feature by
summing (lift - 1) * confidence over every rule it appears in, and compare
against a classic statistical baseline (mutual information).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .rules import Rule


def feature_of(item: str) -> str:
    """'glucose=high' -> 'glucose';  'smoker' -> 'smoker'."""
    return item.split("=", 1)[0]


def rule_based_scores(rules: List[Rule]) -> pd.Series:
    scores: Dict[str, float] = defaultdict(float)
    for r in rules:
        weight = max(r.lift - 1.0, 0.0) * r.confidence
        for item in r.antecedent:
            scores[feature_of(item)] += weight
    s = pd.Series(scores, dtype=float).sort_values(ascending=False)
    return s / s.max() if len(s) and s.max() > 0 else s


def mutual_info_scores(df: pd.DataFrame, feature_cols: Sequence[str], label_cols: Sequence[str],
                       random_state: int = 42) -> pd.Series:
    """Mean mutual information of each feature against every disease label."""
    X = df[list(feature_cols)].copy()
    discrete = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
            discrete.append(True)
        else:
            discrete.append(X[c].nunique() <= 2)
    per_label = [
        mutual_info_classif(X.values, df[lbl].values, discrete_features=np.array(discrete),
                            random_state=random_state)
        for lbl in label_cols
    ]
    s = pd.Series(np.mean(per_label, axis=0), index=X.columns).sort_values(ascending=False)
    return s / s.max() if s.max() > 0 else s


def select_top_k(scores: pd.Series, k: int) -> List[str]:
    return list(scores.head(k).index)
