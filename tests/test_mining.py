import random

import pytest

from dpm import AssociativeClassifier, apriori, fpgrowth, generate_rules

TEXTBOOK = [
    ["bread", "milk"],
    ["bread", "diaper", "beer", "eggs"],
    ["milk", "diaper", "beer", "cola"],
    ["bread", "milk", "diaper", "beer"],
    ["bread", "milk", "diaper", "cola"],
]


def test_apriori_textbook_supports():
    f = apriori(TEXTBOOK, min_support=0.6)
    assert f[frozenset(["bread"])] == pytest.approx(0.8)
    assert f[frozenset(["diaper", "beer"])] == pytest.approx(0.6)
    assert frozenset(["cola"]) not in f


@pytest.mark.parametrize("seed", range(5))
def test_apriori_and_fpgrowth_agree_on_random_data(seed):
    rng = random.Random(seed)
    items = list("abcdefghij")
    tx = [rng.sample(items, rng.randint(1, 6)) for _ in range(200)]
    a = apriori(tx, 0.08)
    b = fpgrowth(tx, 0.08)
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k])


def test_max_len_respected():
    f = fpgrowth(TEXTBOOK, 0.2, max_len=2)
    assert max(len(s) for s in f) == 2


def test_rule_metrics():
    f = apriori(TEXTBOOK, 0.4)
    rules = generate_rules(f, min_confidence=0.0, min_lift=0.0)
    r = next(r for r in rules if r.antecedent == {"diaper"} and r.consequent == {"beer"})
    assert r.confidence == pytest.approx(0.75)
    assert r.lift == pytest.approx(0.75 / 0.6)


def test_consequent_filter_targets_only_labels():
    tx = [["glucose=high", "DX:diabetes"]] * 8 + [["glucose=normal"]] * 12
    rules = generate_rules(apriori(tx, 0.1), 0.5, 1.0, consequent_filter=["DX:diabetes"])
    assert rules and all(r.consequent == {"DX:diabetes"} for r in rules)


def test_associative_classifier_learns_simple_rule():
    tx = [["glucose=high", "DX:diabetes"]] * 30 + [["glucose=normal"]] * 70
    clf = AssociativeClassifier(["diabetes"], min_support=0.1, min_confidence=0.8).fit(tx)
    pred = clf.predict([["glucose=high"], ["glucose=normal"]])
    assert pred[:, 0].tolist() == [1, 0]
    assert clf.explain(["glucose=high"])["diabetes"] is not None
