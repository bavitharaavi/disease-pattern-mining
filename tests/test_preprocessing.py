import pandas as pd

from dpm.preprocessing import discretize, prepare, to_transactions


def test_clinical_bins():
    df = pd.DataFrame({"glucose": [90, 110, 150]})
    assert discretize(df, ["glucose"])["glucose"].tolist() == ["normal", "prediabetic", "high"]


def test_transactions_binary_and_labels():
    df = pd.DataFrame({"smoker": [1, 0], "bmi": ["obese", "normal"], "diabetes": [1, 0]})
    tx = to_transactions(df, label_cols=["diabetes"])
    assert sorted(tx[0]) == ["DX:diabetes", "bmi=obese", "smoker"]
    assert tx[1] == ["bmi=normal"]


def test_prepare_end_to_end():
    df = pd.DataFrame({"glucose": [90, 150, 130, 95], "age": [30, 70, 50, 45], "diabetes": [0, 1, 1, 0]})
    tx = prepare(df, ["diabetes"])
    assert "glucose=high" in tx[1] and "DX:diabetes" in tx[1]
