"""Generate a synthetic multi-disease patient dataset.

Real clinical data (e.g. UCI Heart Disease, Pima Indians Diabetes, Chronic
Kidney Disease) can be dropped into data/ instead. This generator exists so
the pipeline runs end-to-end out of the box with no downloads and no PHI.

Diseases are produced from latent risk scores built on known clinical
relationships (glucose/HbA1c -> diabetes, BP/cholesterol/smoking -> heart
disease, diabetes/BP/creatinine -> CKD), plus noise.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 85, n)
    sex = rng.integers(0, 2, n)  # 1 = male
    bmi = np.clip(rng.normal(27, 5, n), 15, 50)
    smoker = rng.binomial(1, 0.22, n)
    physically_active = rng.binomial(1, 0.45, n)
    family_history = rng.binomial(1, 0.3, n)

    glucose = np.clip(rng.normal(95, 15, n) + 0.9 * (bmi - 25) + 0.3 * (age - 45)
                      + 12 * family_history, 60, 300)
    hba1c = np.clip(4.6 + 0.018 * glucose + rng.normal(0, 0.35, n), 4, 14)
    systolic_bp = np.clip(rng.normal(118, 12, n) + 0.45 * (age - 45) + 0.8 * (bmi - 25)
                          + 6 * smoker, 85, 210)
    cholesterol = np.clip(rng.normal(190, 30, n) + 0.6 * (age - 45) + 10 * smoker
                          - 12 * physically_active, 110, 350)
    max_heart_rate = np.clip(220 - age - rng.normal(25, 12, n) + 10 * physically_active, 70, 200)

    diabetes_risk = 0.09 * (glucose - 126) + 2.2 * (hba1c - 6.5) + 0.8 * family_history
    diabetes = rng.binomial(1, sigmoid(diabetes_risk))

    heart_risk = (0.05 * (systolic_bp - 140) + 0.025 * (cholesterol - 240) + 1.4 * smoker
                  + 0.04 * (age - 60) - 0.9 * physically_active + 0.5 * sex + 0.6 * diabetes - 0.5)
    heart_disease = rng.binomial(1, sigmoid(heart_risk))

    creatinine = np.clip(rng.normal(0.95, 0.2, n) + 0.35 * diabetes
                         + 0.008 * np.maximum(systolic_bp - 130, 0) + 0.006 * (age - 45), 0.4, 6)
    ckd_risk = 3.5 * (creatinine - 1.3) + 0.9 * diabetes + 0.03 * (systolic_bp - 140) - 0.3
    ckd = rng.binomial(1, sigmoid(ckd_risk))

    return pd.DataFrame({
        "age": age, "sex_male": sex, "bmi": bmi.round(1), "smoker": smoker,
        "physically_active": physically_active, "family_history_diabetes": family_history,
        "glucose": glucose.round(0), "hba1c": hba1c.round(1), "systolic_bp": systolic_bp.round(0),
        "cholesterol": cholesterol.round(0), "max_heart_rate": max_heart_rate.round(0),
        "creatinine": creatinine.round(2),
        "diabetes": diabetes, "heart_disease": heart_disease, "ckd": ckd,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/patients.csv")
    a = p.parse_args()
    df = generate(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    prev = df[["diabetes", "heart_disease", "ckd"]].mean().round(3).to_dict()
    print(f"Wrote {len(df)} rows to {a.out}. Prevalence: {prev}")
