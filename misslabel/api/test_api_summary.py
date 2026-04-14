"""
tests/test_api_summary.py
--------------------------
Tests for POST /summary and GET /healthz.

Run with:
    pytest tests/test_api_summary.py -v
"""

from __future__ import annotations

import math
import pytest
from fastapi.testclient import TestClient

from misslabel.api.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post(payload: dict) -> dict:
    r = client.post("/summary", json=payload)
    return r


IRIS_CLEAN  = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
IRIS_NOISY  = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]   # 2 flips: idx 2 (0→1), idx 5 (1→2)


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------

class TestHealthz:
    def test_200(self):
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_body(self):
        r = client.get("/healthz")
        assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Happy-path integer labels
# ---------------------------------------------------------------------------

class TestSummaryIntegerLabels:
    def test_status_200(self):
        r = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY})
        assert r.status_code == 200

    def test_n(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        assert data["n"] == 10

    def test_k(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        assert data["k"] == 3

    def test_n_flipped(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        assert data["n_flipped"] == 2

    def test_flip_rate(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        assert math.isclose(data["flip_rate"], 0.2, rel_tol=1e-6)

    def test_per_class_keys_are_strings(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        for k in data["per_class"]:
            assert isinstance(k, str)

    def test_per_class_n_sum_equals_n(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        total = sum(v["n"] for v in data["per_class"].values())
        assert total == data["n"]

    def test_T_hat_shape(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        T = data["T_hat"]
        assert len(T) == 3
        assert all(len(row) == 3 for row in T)

    def test_T_hat_row_stochastic(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        for row in data["T_hat"]:
            assert math.isclose(sum(row), 1.0, abs_tol=1e-6)

    def test_P_t_absent_when_t_not_given(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        assert data["P_t"] is None
        assert data["max_T_error"] is None

    def test_theoretical_flip_rates_absent_when_t_not_given(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY}).json()
        for stats in data["per_class"].values():
            assert stats["flip_rate_theoretical"] is None


# ---------------------------------------------------------------------------
# Happy-path string labels
# ---------------------------------------------------------------------------

LABELS_STR_TRUE  = ["cat", "cat", "dog", "dog", "bird", "bird"]
LABELS_STR_NOISY = ["cat", "dog", "dog", "dog", "bird", "cat"]   # 2 flips

class TestSummaryStringLabels:
    def test_status_200(self):
        r = post({"y_true": LABELS_STR_TRUE, "y_noisy": LABELS_STR_NOISY})
        assert r.status_code == 200

    def test_k(self):
        data = post({"y_true": LABELS_STR_TRUE, "y_noisy": LABELS_STR_NOISY}).json()
        assert data["k"] == 3

    def test_per_class_keys_are_original_strings(self):
        data = post({"y_true": LABELS_STR_TRUE, "y_noisy": LABELS_STR_NOISY}).json()
        assert set(data["per_class"].keys()) == {"cat", "dog", "bird"}

    def test_n_flipped(self):
        data = post({"y_true": LABELS_STR_TRUE, "y_noisy": LABELS_STR_NOISY}).json()
        assert data["n_flipped"] == 2


# ---------------------------------------------------------------------------
# With t supplied — theoretical quantities present
# ---------------------------------------------------------------------------

class TestSummaryWithT:
    def test_P_t_present(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": 0.5}).json()
        assert data["P_t"] is not None

    def test_P_t_shape(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": 0.5}).json()
        P = data["P_t"]
        assert len(P) == 3 and all(len(row) == 3 for row in P)

    def test_P_t_row_stochastic(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": 0.5}).json()
        for row in data["P_t"]:
            assert math.isclose(sum(row), 1.0, abs_tol=1e-6)

    def test_max_T_error_present(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": 0.5}).json()
        assert data["max_T_error"] is not None
        assert isinstance(data["max_T_error"], float)

    def test_max_T_error_non_negative(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": 0.5}).json()
        assert data["max_T_error"] >= 0.0

    def test_theoretical_flip_rates_present(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": 0.5}).json()
        for stats in data["per_class"].values():
            assert stats["flip_rate_theoretical"] is not None

    def test_t_zero_means_no_flips_theoretical(self):
        """At t=0, P(t)=I so theoretical flip rate should be 0 for all classes."""
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_CLEAN, "t": 0.0}).json()
        for stats in data["per_class"].values():
            assert math.isclose(stats["flip_rate_theoretical"], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# With explicit pi
# ---------------------------------------------------------------------------

class TestSummaryWithPi:
    def test_uniform_pi_accepted(self):
        pi = [1/3, 1/3, 1/3]
        r = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "pi": pi, "t": 0.3})
        assert r.status_code == 200

    def test_non_uniform_pi_accepted(self):
        pi = [0.5, 0.3, 0.2]
        r = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "pi": pi, "t": 0.3})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSummaryEdgeCases:
    def test_no_flips(self):
        data = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_CLEAN}).json()
        assert data["n_flipped"] == 0
        assert math.isclose(data["flip_rate"], 0.0)

    def test_binary_labels(self):
        y = [0, 0, 1, 1, 0, 1]
        data = post({"y_true": y, "y_noisy": y}).json()
        assert data["k"] == 2

    def test_single_class_all_same(self):
        """All labels identical: k=1, no flips possible."""
        y = [7, 7, 7, 7]
        data = post({"y_true": y, "y_noisy": y}).json()
        assert data["k"] == 1
        assert data["n_flipped"] == 0


# ---------------------------------------------------------------------------
# Validation errors — 422
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_length_mismatch(self):
        r = post({"y_true": [0, 1, 2], "y_noisy": [0, 1]})
        assert r.status_code == 422

    def test_empty_y_true(self):
        r = post({"y_true": [], "y_noisy": []})
        assert r.status_code == 422

    def test_negative_t(self):
        r = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "t": -0.1})
        assert r.status_code == 422

    def test_pi_negative_value(self):
        r = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "pi": [-0.1, 0.6, 0.5]})
        assert r.status_code == 422

    def test_pi_does_not_sum_to_one(self):
        r = post({"y_true": IRIS_CLEAN, "y_noisy": IRIS_NOISY, "pi": [0.1, 0.1, 0.1]})
        assert r.status_code == 422

    def test_missing_y_true(self):
        r = post({"y_noisy": IRIS_NOISY})
        assert r.status_code == 422

    def test_missing_y_noisy(self):
        r = post({"y_true": IRIS_CLEAN})
        assert r.status_code == 422
