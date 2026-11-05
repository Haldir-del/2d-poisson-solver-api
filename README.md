# 2D Poisson Solver API

A production-ready HTTP API for the stationary 2D Poisson equation

```
-∇²T = f(x,y)     in Ω = [0, L]²
 T   = g           on ∂Ω            (Dirichlet)
```

Built for **steady-state heat conduction, electrostatics (potential), and
potential flow**. It is a numerical solver only: it does not model transient
problems or Navier–Stokes/CFD.

The default solver is a **direct DST/FFT** method that works with any source
term, returns the exact solution of the discrete system, and needs no
iteration or convergence tuning. Classical PCG, SOR, Jacobi, and Gauss–Seidel
are available for comparison.

> ## Get a production key
> **[Subscribe to 2D Poisson Solver API on RapidAPI →](https://rapidapi.com/)**
>
> RapidAPI handles authentication, quotas, and billing. Use your key with the
> `/solve` endpoint as shown in the Python example below.

---

## Performance

The direct DST/FFT path is `O(N log N)` and solves a `255 × 255` grid
(65,025 unknowns) in about **11 ms** in-process, versus about **789 ms** for
PCG at the same tolerance — roughly **73× faster** on the largest supported
grid.

| Grid `n` | Unknowns | DST/FFT | PCG (`tol=1e-8`) | Speed-up |
|---:|---:|---:|---:|---:|
| 63 | 3,969 | 0.87 ms | 16.7 ms | 19× |
| 127 | 16,129 | 2.72 ms | 159.5 ms | 59× |
| 255 | 65,025 | 10.75 ms | 788.7 ms | 73× |

Measured in-process on a development machine (median of three runs, Gaussian
source, residual ≤ `1e-8`). HTTP/proxy overhead adds a few milliseconds.
Reproduce with:

```bash
python benchmarks/benchmark.py
```

PCG iteration counts grow with the grid (97 → 198 → 401 for the rows above),
so its cost grows faster than the DST solver's. The FFT path always reports
`iterations: 0` and `direct: true`.

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive OpenAPI docs, or
<http://localhost:8000/demo> for a heat-map demo.

```bash
curl -X POST http://localhost:8000/solve \
  -H 'Content-Type: application/json' \
  -d '{"n":255,"source_mode":"gaussian","source_amplitude":100,"method":"fft"}'
```

## Python client (RapidAPI)

```python
import os
import requests

API_KEY = os.environ["RAPIDAPI_KEY"]
API_HOST = os.environ["RAPIDAPI_HOST"]  # e.g. your-listing.p.rapidapi.com

response = requests.post(
    f"https://{API_HOST}/solve",
    headers={
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST,
        "Content-Type": "application/json",
    },
    json={
        "n": 255,
        "source_mode": "gaussian",
        "source_amplitude": 100.0,
        "method": "fft",
        "tolerance": 1e-8,
    },
    timeout=30,
)
response.raise_for_status()

data = response.json()
print(data["success"], data["solve_time_ms"], data["residual"])
print(data["temp_min"], data["temp_max"], data["temp_mean"])

# The field is indexed as field[j][i] (j = y, i = x).
field = data["field"]
```

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | public | Health/readiness, uptime, configured port |
| `GET /methods` | public | Methods, limits, and references |
| `POST /solve` | RapidAPI proxy secret | Full solve |
| `POST /demo/solve` | public | Capped solve for the demo page |
| `GET /demo` | public | Interactive HTML demo |
| `GET /docs` | public | OpenAPI/Swagger UI |

In production, set `RAPIDAPI_PROXY_SECRET`. RapidAPI injects the matching
`X-RapidAPI-Proxy-Secret` header, so direct traffic to the Railway URL is
rejected with `403`. Locally, leave it unset to disable the check.

`/demo/solve` is public but capped at `n ≤ 64`, `max_iter ≤ 2000`, and
`tolerance ≥ 1e-6`.

## Request fields

All numeric fields reject `NaN` and `Inf`; every field is bounded.

| Field | Type | Default | Bounds / notes |
|---|---|---|---|
| `n` | int | `64` | `3`–`256` nodes per side (`n²` unknowns) |
| `length` | float | `1.0` | `1e-6`–`1e6`, domain side length |
| `boundary` | float | `0.0` | `-1e6`–`1e6`, Dirichlet value on all boundaries |
| `source_mode` | string | `"gaussian"` | `constant` \| `gaussian` \| `sine` |
| `source_amplitude` | float | `100.0` | `-1e6`–`1e6` |
| `method` | string | `"fft"` | `fft` \| `pcg` \| `sor` \| `jacobi` \| `gauss_seidel` |
| `omega` | float or null | `null` | SOR relaxation; `null` = analytical optimum |
| `tolerance` | float | `1e-8` | Relative residual, `1e-10`–`1e-2` |
| `max_iter` | int | `10000` | `1`–`20000` |
| `heat_source_x` | float | `0.5` | Gaussian center X, normalized `[0,1]` |
| `heat_source_y` | float | `0.5` | Gaussian center Y, normalized `[0,1]` |
| `heat_sigma` | float | `0.08` | Gaussian width, normalized `1e-3`–`1e6` |

## Response

```json
{
  "success": true,
  "method": "fft",
  "direct": true,
  "grid_n": 255,
  "iterations": 0,
  "residual": 1.38e-12,
  "solve_time_ms": 10.75,
  "temp_min": 0.0,
  "temp_max": 3.42,
  "temp_mean": 0.81,
  "field": [[0.0, "..."]],
  "error": null
}
```

- `success` is `true` only when the residual is finite and
  `residual ≤ tolerance × 1.01`.
- `direct` is `true` for `fft`.
- `error` distinguishes tolerance, iteration-cap, and time-limit failures.
- Invalid input (including `Inf`/`NaN`) returns `422`, never `500`.

## Methods

| Method | Complexity | Recommended use |
|---|---|---|
| `fft` **(default)** | `O(N log N)` | Any source and boundary. Direct DST solve, no iteration. |
| `pcg` | ~`O(N^{3/2})` | Iterative comparison; Jacobi preconditioning is a no-op on this uniform grid. |
| `sor` | ~`O(N^{3/2})` | Classical SOR with analytical optimal `ω = 2/(1+sin(π/(n+1)))`. |
| `jacobi` | `O(N²)` iterations | Comparison / small grids. |
| `gauss_seidel` | `O(N²)` iterations | Red-black Gauss–Seidel, ~2× faster than Jacobi. |

Iterative methods run under a wall-clock budget
(`POISSON_TIME_LIMIT_S`, default `10` seconds) and report a clear error when
they exceed it.

## Numerical notes

- 5-point finite differences, second-order `O(h²)`.
- Residual is reported as `‖Ax − b‖ / ‖b‖`, not just an iteration count.
- Verified against manufactured solutions (observed convergence rate ≈ 2.0)
  and against PCG on the same grid (max difference `< 1e-8`).
- Maximum request: `256² = 65,536` unknowns, about `1.3 MB` of JSON.
- The DST diagonalizes the discrete Dirichlet Laplacian for **any** right-hand
  side; sinusoidal sources are not required.

## Deploy on Railway

The repository includes a multi-stage `Dockerfile` and `railway.toml`.

1. Push the repository to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**.
3. Railway detects `railway.toml` / `Dockerfile` and builds automatically.
4. Add environment variables:
   - `RAPIDAPI_PROXY_SECRET` — same value you configure in the RapidAPI Hub.
   - `POISSON_TIME_LIMIT_S` — optional, default `10`.
   - `PORT` is injected by Railway; the container falls back to `8000` locally.
5. Set the health check path to `/health` (already configured).
6. Copy the public Railway URL and point RapidAPI at it.

`GET /health` returns the effective port, and the Docker `HEALTHCHECK` also
reads `PORT`, so the service works with either Railway's assigned port or the
local `8000` default.

See [`RAPIDAPI_DEPLOY.md`](./RAPIDAPI_DEPLOY.md) for the full RapidAPI
publication checklist.

## Tests

```bash
python -m pytest test_api.py -q
```

21 tests cover the default FFT path, FFT-vs-PCG agreement, non-zero
boundaries, Inf/out-of-range validation, tolerance semantics, convergence
order, the RapidAPI secret gate, and the `/demo/solve` caps. The same suite
runs in GitHub Actions on every push and pull request
(`.github/workflows/tests.yml`).

## Project layout

```
.
├── main.py                 # FastAPI app, auth, demo
├── models.py               # Pydantic request/response models
├── solver.py               # DST/FFT + iterative numerical core
├── test_api.py             # regression tests
├── benchmarks/benchmark.py # reproducible FFT vs PCG benchmark
├── .github/workflows/tests.yml # GitHub Actions test workflow
├── Dockerfile              # multi-stage runtime image
├── railway.toml            # Railway build/healthcheck config
├── .env.example            # environment variable template
├── .gitignore
├── .dockerignore
├── README.md
└── RAPIDAPI_DEPLOY.md
```

## References

- Yang & Gobbert (2009), optimal SOR relaxation for 2D Poisson.
- MIT 18.336, discrete sine transforms for Dirichlet Poisson problems.
- Higham (2002), *Accuracy and Stability of Numerical Algorithms*.
- Demmel, CS267 lecture notes.
- Nocedal & Wright, *Numerical Optimization*.
