import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_default_method_is_fft():
    r = client.post("/solve", json={})
    assert r.status_code == 200
    j = r.json()
    assert j["method"] == "fft"
    assert j["direct"] is True
    assert j["success"] is True
    assert j["iterations"] == 0
    assert len(j["field"]) == 64


def test_solve_gaussian_default_fft():
    r = client.post("/solve", json={"n": 32, "source_mode": "gaussian"})
    assert r.status_code == 200
    j = r.json()
    assert j["success"] and j["residual"] < 1e-8
    assert len(j["field"]) == 32
    assert j["temp_max"] > j["temp_min"]


@pytest.mark.parametrize("mode", ["gaussian", "constant", "sine"])
def test_fft_accepts_any_source(mode):
    r = client.post("/solve", json={"n": 32, "method": "fft", "source_mode": mode})
    assert r.status_code == 200
    j = r.json()
    assert j["success"] and j["residual"] < 1e-8


def test_fft_matches_pcg_gaussian():
    payload = {
        "n": 32,
        "source_mode": "gaussian",
        "source_amplitude": 100.0,
        "tolerance": 1e-10,
    }
    a = client.post("/solve", json={**payload, "method": "fft"}).json()
    b = client.post("/solve", json={**payload, "method": "pcg"}).json()
    assert a["success"] and b["success"]
    diff = np.max(np.abs(np.array(a["field"]) - np.array(b["field"])))
    assert diff < 1e-8


def test_solve_gaussian_pcg():
    r = client.post("/solve", json={"n": 32, "method": "pcg", "source_mode": "gaussian"})
    assert r.status_code == 200
    j = r.json()
    assert j["success"]
    assert j["residual"] < 1e-6
    assert len(j["field"]) == 32
    assert j["temp_max"] > j["temp_min"]


def test_solve_fft_nonzero_boundary():
    r = client.post(
        "/solve",
        json={
            "n": 32,
            "method": "fft",
            "source_mode": "constant",
            "source_amplitude": 0,
            "boundary": 42.0,
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert abs(j["temp_mean"] - 42.0) < 1e-6
    assert j["success"]


def test_solve_small_n():
    r = client.post(
        "/solve",
        json={"n": 3, "method": "sor", "source_mode": "constant", "source_amplitude": 10},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["success"]
    assert len(j["field"]) == 3


def test_solve_jacobi_small():
    r = client.post(
        "/solve",
        json={"n": 16, "method": "jacobi", "source_mode": "constant", "source_amplitude": 10},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["success"] and j["residual"] < 1e-6


def test_solve_boundary_nonzero():
    r = client.post(
        "/solve",
        json={
            "n": 32,
            "method": "pcg",
            "source_mode": "constant",
            "source_amplitude": 0,
            "boundary": 42.0,
        },
    )
    assert r.status_code == 200
    j = r.json()
    # With f=0 and boundary=42, the exact solution is T=42 everywhere.
    assert abs(j["temp_mean"] - 42.0) < 1e-6
    assert j["success"]


def test_solve_invalid_n_422():
    r = client.post("/solve", json={"n": 10000})
    assert r.status_code == 422


def test_missing_body_422():
    r = client.post("/solve")
    assert r.status_code == 422


def test_empty_json_uses_defaults():
    r = client.post("/solve", json={})
    assert r.status_code == 200
    assert r.json()["method"] == "fft"


def test_rejects_nonfinite():
    for raw in ['{"source_amplitude": 1e309}', '{"boundary": -1e309}']:
        r = client.post("/solve", content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 422, raw


def test_rejects_out_of_range():
    for payload in [
        {"source_amplitude": 1e12},
        {"boundary": 1e12},
        {"length": 1e-9},
        {"heat_sigma": 1e-9},
        {"tolerance": 1e-14},
        {"max_iter": 50000},
    ]:
        r = client.post("/solve", json=payload)
        assert r.status_code == 422, payload


def test_success_false_when_tolerance_not_met():
    r = client.post(
        "/solve",
        json={
            "n": 64,
            "method": "jacobi",
            "source_mode": "gaussian",
            "max_iter": 5,
            "tolerance": 1e-8,
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["success"] is False
    assert j["error"]


def test_convergence_order():
    errors = []
    for n in (15, 31, 63):
        r = client.post(
            "/solve",
            json={
                "n": n,
                "method": "fft",
                "source_mode": "sine",
                "source_amplitude": 2 * math.pi**2,
                "tolerance": 1e-10,
            },
        )
        assert r.status_code == 200
        T = np.array(r.json()["field"])
        h = 1.0 / (n + 1)
        x = np.linspace(h, 1 - h, n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
        errors.append(float(np.max(np.abs(T - exact))))

    rates = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.8 < rate < 2.2 for rate in rates), (errors, rates)


def test_rapidapi_proxy_secret(monkeypatch):
    payload = {"n": 16, "method": "fft"}
    monkeypatch.setenv("RAPIDAPI_PROXY_SECRET", "s3cret")

    r = client.post("/solve", json=payload)
    assert r.status_code == 403

    r = client.post("/solve", json=payload, headers={"X-RapidAPI-Proxy-Secret": "wrong"})
    assert r.status_code == 403

    r = client.post("/solve", json=payload, headers={"X-RapidAPI-Proxy-Secret": "s3cret"})
    assert r.status_code == 200
    assert r.json()["success"]


def test_demo_solve_is_limited_and_public(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_PROXY_SECRET", "s3cret")
    r = client.post("/demo/solve", json={"n": 128, "method": "fft"})
    assert r.status_code == 200
    j = r.json()
    assert j["grid_n"] == 64
    assert j["success"]
