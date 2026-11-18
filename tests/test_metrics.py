"""Unit tests with hand-computable values. Run: python -m pytest -q tests  (or: python tests/test_metrics.py)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factbench import metrics as M, analysis as A, data


def close(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


def test_ece_perfect_and_worst():
    close(M.ece([0.5] * 4, [1, 1, 0, 0]), 0.0)
    close(M.ece([1.0] * 4, [0, 0, 0, 0]), 1.0)


def test_brier_known_values():
    close(M.brier(np.eye(4)[[0, 1]], np.array([0, 1])), 0.0)
    close(M.brier(np.full((1, 4), 0.25), np.array([2])), 0.75)


def test_auroc():
    close(M.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
    close(M.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
    close(M.auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)
    assert np.isnan(M.auroc([0.1, 0.2], [1, 1]))


def test_risk_coverage_and_aurc():
    conf, cor = [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0]
    cov, risk = M.risk_coverage(conf, cor)
    np.testing.assert_allclose(risk, [0, 0, 1 / 3, 0.5])
    close(M.aurc(conf, cor), (0 + 0 + 1 / 3 + 0.5) / 4)
    close(M.selective_accuracy(conf, cor, 0.5), 1.0)


def test_consistency():
    frac, mode = M.consistency(np.array([[1, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 3]]))
    np.testing.assert_allclose(frac, [1.0, 0.5, 0.25])
    assert mode[0] == 1 and mode[1] == 1


def test_canonicalize_roundtrip():
    perms = [[0, 1, 2], [2, 0, 1]]            # variant 1: display A->opt2, B->opt0, C->opt1
    logp = np.log([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])  # variant 1 favours display A => option 2
    P = A.canonicalize(perms, logp)
    np.testing.assert_allclose(P[0], [0.1, 0.2, 0.7])
    np.testing.assert_allclose(P[1], [0.3, 0.1, 0.6])
    assert P[1].argmax() == 2 and abs(P.sum(1) - 1).max() < 1e-12


def test_holm_and_mcnemar():
    np.testing.assert_allclose(M.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    b, c, p = A.mcnemar_exact([1, 1, 1, 0, 0], [0, 0, 0, 0, 1])
    assert (b, c) == (3, 1) and 0 < p <= 1


def test_stratified_sample_is_deterministic_and_sized():
    labels = ["yes"] * 55 + ["no"] * 34 + ["maybe"] * 11
    a, b = data.stratified_sample(labels, 50, 42), data.stratified_sample(labels, 50, 42)
    assert a == b and len(a) == 50


if __name__ == "__main__":   # fallback runner when pytest is unavailable
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
