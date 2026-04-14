"""
misslabel.api.app
-----------------
FastAPI application exposing misslabel diagnostics over HTTP.

Routes
------
GET  /healthz          liveness check
POST /summary          audit noisy vs clean label vectors
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from misslabel import summary as _summary

from .schemas import SummaryRequest, SummaryResponse, PerClassStats

app = FastAPI(
    title="misslabel API",
    description=(
        "HTTP interface to the misslabel F81 noise-auditing library. "
        "See https://github.com/cassiusma/misslabel for the noise model details."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_ndarray(labels: list) -> np.ndarray:
    """Convert a list of str/int labels to a numpy array."""
    return np.array(labels)


def _matrix_to_list(arr: np.ndarray | None) -> list | None:
    """Convert a 2-D numpy array to a nested Python list, or return None."""
    if arr is None:
        return None
    return arr.tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["meta"])
def healthz() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post(
    "/summary",
    response_model=SummaryResponse,
    tags=["audit"],
    summary="Audit a noisy label vector against its clean counterpart.",
    response_description="Empirical and (optionally) theoretical noise statistics.",
)
def post_summary(req: SummaryRequest) -> SummaryResponse:
    """
    Compare **y_true** (clean) and **y_noisy** (corrupted) label vectors and
    return per-class flip rates, the empirical transition matrix **T̂**, and —
    when `t` is supplied — the theoretical F81 transition matrix **P(t)** with
    the max elementwise error `max|T̂ − P(t)|`.

    ### Parameters
    - **y_true** – clean label vector (strings or integers).
    - **y_noisy** – noisy label vector, same length as `y_true`.
    - **pi** *(optional)* – stationary distribution used during injection.
      Estimated from `y_true` if omitted.
    - **t** *(optional)* – F81 time parameter used during injection.
      When provided, theoretical quantities are included in the response.
    """
    try:
        report = _summary(
            y_true=_to_ndarray(req.y_true),
            y_noisy=_to_ndarray(req.y_noisy),
            pi=np.array(req.pi) if req.pi is not None else None,
            t=req.t,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Normalise per_class keys to strings (JSON requires string keys)
    per_class = {
        str(cls): PerClassStats(**stats)
        for cls, stats in report["per_class"].items()
    }

    return SummaryResponse(
        n=report["n"],
        k=report["k"],
        n_flipped=report["n_flipped"],
        flip_rate=report["flip_rate"],
        per_class=per_class,
        T_hat=_matrix_to_list(report["T_hat"]),
        P_t=_matrix_to_list(report["P_t"]),
        max_T_error=report["max_T_error"],
    )
