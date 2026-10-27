"""End-to-end experiment: mine patterns, select features, detect diseases.

Usage:
    python -m dpm.pipeline --data data/patients.csv --labels diabetes heart_disease ckd
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .apriori import apriori
from .classifier import AssociativeClassifier
from .feature_selection import mutual_info_scores, rule_based_scores, select_top_k
from .fpgrowth import fpgrowth
from .preprocessing import prepare
from .rules import generate_rules, prune_redundant


def benchmark_miners(transactions, min_support, max_len):
    timings = {}
    results = {}
    for name, fn in (("apriori", apriori), ("fpgrowth", fpgrowth)):
        t0 = time.perf_counter()
        results[name] = fn(transactions, min_support, max_len)
        timings[name] = round(time.perf_counter() - t0, 4)
    assert set(results["apriori"]) == set(results["fpgrowth"]), "miners disagree"
    return results["fpgrowth"], timings


def multilabel_metrics(y_true, y_pred, labels):
    out = {}
    for j, lbl in enumerate(labels):
        out[lbl] = {
            "precision": round(precision_score(y_true[:, j], y_pred[:, j], zero_division=0), 4),
            "recall": round(recall_score(y_true[:, j], y_pred[:, j], zero_division=0), 4),
            "f1": round(f1_score(y_true[:, j], y_pred[:, j], zero_division=0), 4),
        }
    out["macro_f1"] = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
    return out


def run(data: str, labels, min_support=0.05, min_confidence=0.6, min_lift=1.2,
        max_len=4, top_k=6, out_dir="results", seed=42):
    df = pd.read_csv(data)
    feature_cols = [c for c in df.columns if c not in labels]
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=seed,
                                         stratify=df[labels[0]])
    train_tx = prepare(train_df, labels)
    test_tx = prepare(test_df, labels)

    # 1) Frequent pattern mining (+ sanity check that both algorithms agree)
    frequent, timings = benchmark_miners(train_tx, min_support, max_len)
    print(f"[mine] {len(frequent)} frequent itemsets | time(s): {timings}")

    # 2) Disease-targeted association rules
    rules = prune_redundant(generate_rules(frequent, min_confidence, min_lift,
                                           consequent_filter=[f"DX:{l}" for l in labels]))
    print(f"[rules] {len(rules)} non-redundant disease rules. Top 10:")
    for r in rules[:10]:
        print("   ", r)

    # 3) Feature selection: rule-based vs mutual information
    rb = rule_based_scores(rules)
    mi = mutual_info_scores(train_df, feature_cols, labels, seed)
    selected = [f for f in select_top_k(rb, top_k) if f in feature_cols]
    print(f"[features] rule-based top-{top_k}: {selected}")

    # 4) Detection: interpretable associative classifier vs RF baselines
    y_train, y_test = train_df[labels].values, test_df[labels].values
    ac = AssociativeClassifier(labels, min_support, min_confidence, min_lift, max_len).fit(train_tx)
    ac_pred = ac.predict(test_tx, threshold=0.5)

    rf_all = RandomForestClassifier(300, random_state=seed, class_weight="balanced")
    rf_all.fit(train_df[feature_cols], y_train)
    rf_sel = RandomForestClassifier(300, random_state=seed, class_weight="balanced")
    rf_sel.fit(train_df[selected], y_train)

    metrics = {
        "associative_classifier": multilabel_metrics(y_test, ac_pred, labels),
        f"random_forest_all_{len(feature_cols)}_features": multilabel_metrics(
            y_test, rf_all.predict(test_df[feature_cols]), labels),
        f"random_forest_selected_{len(selected)}_features": multilabel_metrics(
            y_test, rf_sel.predict(test_df[selected]), labels),
    }
    for name, m in metrics.items():
        print(f"[eval] {name}: macro-F1 = {m['macro_f1']}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in rules]).to_csv(out / "rules.csv", index=False)
    pd.DataFrame({"rule_based": rb, "mutual_info": mi}).fillna(0).round(4).to_csv(
        out / "feature_scores.csv")
    (out / "metrics.json").write_text(json.dumps(
        {"mining_time_s": timings, "n_frequent_itemsets": len(frequent), "n_rules": len(rules),
         "selected_features": selected, "metrics": metrics}, indent=2))
    print(f"[done] results written to {out}/")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/patients.csv")
    p.add_argument("--labels", nargs="+", default=["diabetes", "heart_disease", "ckd"])
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--min-lift", type=float, default=1.2)
    p.add_argument("--max-len", type=int, default=4)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.data, a.labels, a.min_support, a.min_confidence, a.min_lift, a.max_len, a.top_k, a.out)


if __name__ == "__main__":
    main()
