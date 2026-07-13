import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_engine import evaluate_model_reliability


def test_constant_probability_model_is_not_reliable():
    y = np.array([0, 1] * 150, dtype=float)
    p = np.full_like(y, 0.5)

    result = evaluate_model_reliability(y, p)

    assert result["reliable"] is False
    assert "AUC<0.53" in result["reason"]
    assert "BRIER_NO_GAIN" in result["reason"]


def test_discriminating_calibrated_model_can_pass():
    y = np.array([0] * 200 + [1] * 200, dtype=float)
    p = np.array([0.15] * 200 + [0.85] * 200, dtype=float)

    result = evaluate_model_reliability(y, p)

    assert result["reliable"] is True
    assert result["reason"] == "PASS"
    assert result["auc"] == 1.0
