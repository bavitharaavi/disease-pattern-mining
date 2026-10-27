# Frequent Pattern Extraction for Multiple Disease Detection

Mining frequent patterns and association rules from patient records to detect
**diabetes, heart disease and chronic kidney disease (CKD)**, and using those
rules to drive **feature selection** and an **interpretable rule-based classifier**.

**Skills:** Data Mining · Association Rule Mining · Algorithm Optimization · Python

---

## What it does

```
patients.csv ─► discretize (clinical cut-offs) ─► transactions
              ─► Apriori / FP-Growth ─► frequent itemsets
              ─► disease-targeted association rules (+ redundancy pruning)
              ─► rule-based feature ranking  ─► Random Forest on selected features
              ─► Associative classifier (every prediction explained by a rule)
```

| Stage | Module | Notes |
|---|---|---|
| Preprocessing | `src/dpm/preprocessing.py` | Numeric vitals → clinical bins (e.g. glucose `<100 / 100–125 / ≥126`); binary flags → items |
| Apriori | `src/dpm/apriori.py` | From scratch, with join + subset-pruning |
| FP-Growth | `src/dpm/fpgrowth.py` | From scratch FP-tree with conditional-tree recursion — the "algorithm optimization" part |
| Rules | `src/dpm/rules.py` | Support, confidence, lift, leverage, conviction; consequent restricted to diseases; redundant-rule pruning |
| Feature selection | `src/dpm/feature_selection.py` | Scores features by Σ (lift − 1)·confidence over rules; compared with mutual information |
| Classifier | `src/dpm/classifier.py` | CBA-style multi-label associative classifier with `explain()` |

Both miners are implemented without external mining libraries, and the pipeline
asserts they produce identical itemsets on every run.

## Quick start

```bash
git clone https://github.com/<your-username>/disease-pattern-mining.git
cd disease-pattern-mining
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/generate_synthetic_data.py      # writes data/patients.csv
python -m dpm.pipeline                          # mine, select features, evaluate
pytest                                          # 13 unit tests
```

Outputs land in `results/`: `rules.csv`, `feature_scores.csv`, `metrics.json`.

### Example rules (synthetic data)

```
{creatinine=high, glucose=high}              => {DX:diabetes}       conf=0.95  lift=3.46
{glucose=high, hba1c=high}                   => {DX:diabetes}       conf=0.90  lift=3.28
{age=senior, smoker}                         => {DX:heart_disease}  conf=0.69  lift=3.61
```

### Using a real dataset

Any CSV works as long as disease columns are 0/1:

```bash
python -m dpm.pipeline --data data/my_dataset.csv --labels diabetes hypertension \
       --min-support 0.04 --min-confidence 0.65 --min-lift 1.3
```

Good public sources: UCI Heart Disease, Pima Indians Diabetes, UCI Chronic Kidney Disease.
Add cut-offs for new numeric columns in `CLINICAL_BINS` (otherwise they're split into tertiles).

## Using the library

```python
from dpm import fpgrowth, generate_rules, AssociativeClassifier
from dpm.preprocessing import prepare

tx = prepare(df, label_cols=["diabetes", "heart_disease"])
itemsets = fpgrowth(tx, min_support=0.05)
rules = generate_rules(itemsets, 0.6, 1.2, consequent_filter=["DX:diabetes", "DX:heart_disease"])

clf = AssociativeClassifier(["diabetes", "heart_disease"]).fit(tx)
clf.explain(tx[0])   # which rule flagged this patient
```

## Results

Run `python -m dpm.pipeline` and record your numbers here (they depend on the
dataset and thresholds).

| Model | Features | Macro-F1 |
|---|---|---|
| Associative classifier | rules | _fill in_ |
| Random Forest | all | _fill in_ |
| Random Forest | rule-selected | _fill in_ |

## Project structure

```
disease-pattern-mining/
├── data/                 # patients.csv (synthetic) or your own data
├── scripts/generate_synthetic_data.py
├── src/dpm/              # library code
├── tests/                # pytest suite
├── results/              # generated (git-ignored)
├── pyproject.toml
└── requirements.txt
```

## Disclaimer

Research/educational project. Not a medical device and not for clinical decisions.

## License

MIT
