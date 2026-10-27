"""Turn a tabular medical dataset into transactions of categorical items.

Pattern mining works on discrete items ("glucose=high"), so numeric
clinical measurements are binned using clinically meaningful cut-offs
when known, otherwise by quantiles.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Clinically motivated bins: (edges, labels). Edges are inclusive-left.
CLINICAL_BINS: Dict[str, tuple] = {
    "age": ([0, 40, 60, np.inf], ["young", "middle", "senior"]),
    "bmi": ([0, 18.5, 25, 30, np.inf], ["under", "normal", "over", "obese"]),
    "glucose": ([0, 100, 126, np.inf], ["normal", "prediabetic", "high"]),
    "systolic_bp": ([0, 120, 140, np.inf], ["normal", "elevated", "high"]),
    "cholesterol": ([0, 200, 240, np.inf], ["desirable", "borderline", "high"]),
    "hba1c": ([0, 5.7, 6.5, np.inf], ["normal", "prediabetic", "high"]),
    "creatinine": ([0, 1.3, np.inf], ["normal", "high"]),
    "max_heart_rate": ([0, 120, 160, np.inf], ["low", "moderate", "high"]),
}


def discretize(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    n_quantiles: int = 3,
    bins: Optional[Dict[str, tuple]] = None,
) -> pd.DataFrame:
    """Return a copy of *df* where every numeric column is replaced by a bin label."""
    bins = {**CLINICAL_BINS, **(bins or {})}
    out = df.copy()
    numeric_cols = list(numeric_cols) if numeric_cols is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2
    ]
    for col in numeric_cols:
        if col in bins:
            edges, labels = bins[col]
            out[col] = pd.cut(df[col], bins=edges, labels=labels, right=False).astype(str)
        else:
            labels = ["low", "mid", "high"] if n_quantiles == 3 else [f"q{i+1}" for i in range(n_quantiles)]
            out[col] = pd.qcut(df[col], q=n_quantiles, labels=labels, duplicates="drop").astype(str)
    return out


def to_transactions(
    df: pd.DataFrame,
    label_cols: Sequence[str] = (),
    positive_only_binary: bool = True,
) -> List[List[str]]:
    """Convert a discretized dataframe into a list of item lists.

    Binary columns (0/1, yes/no) contribute an item only when positive, e.g.
    "smoker" instead of "smoker=1" / "smoker=0" — this keeps the item space
    small and the rules readable. Disease label columns become items like
    "DX:diabetes".
    """
    label_cols = set(label_cols)
    transactions: List[List[str]] = []
    binary_cols = {
        c for c in df.columns
        if set(pd.unique(df[c].dropna())) <= {0, 1, True, False, "0", "1", "yes", "no"}
    }
    for _, row in df.iterrows():
        items: List[str] = []
        for col, val in row.items():
            if pd.isna(val) or val == "nan":
                continue
            if col in label_cols:
                if str(val).lower() in {"1", "true", "yes"}:
                    items.append(f"DX:{col}")
                continue
            if col in binary_cols and positive_only_binary:
                if str(val).lower() in {"1", "true", "yes"}:
                    items.append(col)
                continue
            items.append(f"{col}={val}")
        transactions.append(items)
    return transactions


def prepare(df: pd.DataFrame, label_cols: Sequence[str]) -> List[List[str]]:
    """Convenience: discretize then convert to transactions."""
    feature_numeric = [
        c for c in df.columns
        if c not in label_cols and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2
    ]
    return to_transactions(discretize(df, feature_numeric), label_cols)
