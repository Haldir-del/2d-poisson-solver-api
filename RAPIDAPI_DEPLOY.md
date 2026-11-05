# 2D Poisson Solver API — RapidAPI Deployment Guide

Stationary solver for `-∇²T = f(x,y)` on a square domain `[0,L]²` with a
Dirichlet boundary condition `T = boundary`. Intended for steady-state heat
conduction, electrostatics, and potential flow. It does not model transient
phenomena, Navier–Stokes, or general CFD.

Use **2D Poisson Solver API** as the public API name and listing title.

## API Reference

### Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | public | Container health check (RapidAPI readiness) |
| `GET /methods` | public | Available methods, limits, references |
| `POST /solve` | proxy secret¹ | Solve the Poisson equation |
| `POST /demo/solve` | public | Capped solve used by the demo page |
| `GET /demo` | public | Interactive HTML demo |

¹ RapidAPI sends `X-RapidAPI-Proxy-Secret` on every proxied request. When the
container's `RAPIDAPI_PROXY_SECRET` env var is set, `/solve` returns `403` if
the header is missing or wrong. This prevents direct use of the container URL
and bypassing RapidAPI billing. Leave the variable unset only for local
development; always set it in production.

The public `/demo/solve` endpoint is deliberately limited (`n ≤ 64`,
`max_iter ≤ 2000`, `tolerance ≥ 1e-6`) so the HTML demo keeps working without
exposing the full solver for free.

### `POST /solve` request body (`PoissonRequest`)

All float fields reject `NaN`/`Inf`; bounds are enforced by Pydantic.

| Field | Type | Default | Range / Notes |
|---|---|---|---|
| `n` | int | `64` | `3`–`256` grid nodes per side (NxN) |
| `length` | float | `1.0` | `1e-6`–`1e6`, domain side length |
| `boundary` | float | `0.0` | `-1e6`–`1e6`, Dirichlet value on all boundaries |
| `source_mode` | string | `"gaussian"` | `"constant"` \| `"gaussian"` \| `"sine"` |
| `source_amplitude` | float | `100.0` | `-1e6`–`1e6` |
| `method` | string | `"fft"` | `"fft"` \| `"pcg"` \| `"sor"` \| `"jacobi"` \| `"gauss_seidel"` |
| `omega` | float or null | `null` | `0.1`–`1.99`; `null`/omitted = analytical optimal ω |
| `tolerance` | float | `1e-8` | `1e-10`–`1e-2` (relative residual) |
| `max_iter` | int | `10000` | `1`–`20000` |
| `heat_source_x` | float | `0.5` | Gaussian center X, normalized `[0,1]` |
| `heat_source_y` | float | `0.5` | Gaussian center Y, normalized `[0,1]` |
| `heat_sigma` | float | `0.08` | Gaussian sigma, normalized, `1e-3`–`1e6` |

### Response (`PoissonResponse`)

`success`, `method`, `direct`, `grid_n`, `iterations`, `residual`,
`solve_time_ms`, `temp_min`, `temp_max`, `temp_mean`, `field` (NxN list),
`error`.

- `success` requires a finite residual **and**
  `residual <= tolerance * 1.01`.
- `direct` is `true` for `fft`; iterative methods report `false`.
- `fft` reports `iterations: 0`.
- `error` distinguishes iteration-cap, time-limit, and tolerance failures.

## Methods

| Method | Complexity | When to use |
|---|---|---|
| `fft` **(default)** | O(N log N) | Direct DST solve for **any** source mode and boundary value. Exact for the discrete system. |
| `pcg` | ~O(N^{3/2}) | Iterative comparison; Jacobi preconditioning is a no-op on the uniform grid |
| `sor` | ~O(N^{3/2}) | Classical SOR with analytical optimal ω when `omega` is omitted |
| `jacobi` | O(N²) iterations | Comparison / small grids only |
| `gauss_seidel` | O(N²) iterations | Red-black Gauss–Seidel, ~2× Jacobi |

Iterative methods run under a hard wall-clock budget:
`POISSON_TIME_LIMIT_S` (default `10` seconds). Requests that exceed it return
`success: false` with an explanatory `error` instead of hanging.

## Numerical stability

- 5-point discretization, `O(h²)`; convergence rate verified against
  manufactured solutions (≈ 2.0).
- Direct DST solution validated against PCG; max difference `< 1e-8`.
- `n` in `[3, 256]` → max 65,536 unknowns per request (RapidAPI-safe payload).
- Relative residual returned, not just an iteration count.
- Non-finite inputs are rejected with `422` before reaching the solver.

## Deploy on Railway

1. Push the repository to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**.
3. Railway reads `railway.toml`, builds the `Dockerfile`, and exposes the
   service on a public domain.
4. Add the environment variables:
   - `RAPIDAPI_PROXY_SECRET` — the value you will configure in RapidAPI.
   - `POISSON_TIME_LIMIT_S` — optional, default `10`.
   - Do **not** hard-code `PORT`; Railway injects it.
5. Confirm `GET https://<railway-domain>/health` returns `200` and reports the
   effective `port`.
6. Keep the Railway URL private. RapidAPI will call it using
   `X-RapidAPI-Proxy-Secret`.

The container command is
`uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1`, so the
same image works on Railway (assigned `PORT`) and locally (`8000`). The
`HEALTHCHECK` reads `PORT` too, and `railway.toml` points Railway at
`/health`.

## Build & publish Docker image

```bash
# Build the image (run from this directory)
docker build -t poisson-solver-api:latest .

# Quick smoke test locally (PORT defaults to 8000)
docker run --rm -p 8000:8000 poisson-solver-api:latest
curl http://localhost:8000/health
curl http://localhost:8000/methods
curl -X POST http://localhost:8000/solve \
  -H 'Content-Type: application/json' \
  -d '{"n":64,"source_mode":"gaussian","source_amplitude":100,"method":"fft","tolerance":1e-8}'
```

The container listens on `${PORT:-8000}`, so the platform-assigned port is
honored automatically.

### RapidAPI listing copy

**Name:** 2D Poisson Solver API

**Short description:** Solve the 2D Poisson equation on a square grid with a
direct DST/FFT solver for steady-state heat conduction, electrostatics, and
potential flow.

**Long description:** A stationary 2D Poisson solver for `-∇²T = f(x,y)` with
Dirichlet boundaries. The default DST/FFT method accepts any source term and
returns the exact solution of the discrete system without iteration. PCG,
SOR, Jacobi, and Gauss–Seidel are available for comparison. Includes strict
input validation, bounded grid sizes, and clear residual reporting. This API
models steady-state problems only; it does not model transient phenomena or
Navier–Stokes/CFD.

### Upload to RapidAPI Hub

1. Go to the RapidAPI Hub → **Create API** → **Docker** (Container) type.
2. Upload the Docker image (Docker Hub / GitHub / registry URL) **or** use the
   "Build from Dockerfile" flow if your Hub role allows it.
3. Set the container port to the platform-provided `PORT` (the image falls
   back to `8000`).
4. Wait for `GET /health` to return `200`.
5. Define the endpoints in the Hub: `POST /solve`, `GET /methods`,
   `GET /health` (and optionally `GET /demo`).
6. Set the request schema from the `PoissonRequest` table above; `field` in
   the response is `number[][]` (NxN).
7. Configure `RAPIDAPI_PROXY_SECRET` in the container environment. RapidAPI
   injects the matching header; direct container traffic is then rejected.
8. Optionally set `POISSON_TIME_LIMIT_S` to tune the iterative budget.

### Local run without Docker

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Tests

```bash
python -m pytest test_api.py -q
```

Twenty-one regression tests cover the default FFT path, FFT vs PCG agreement,
non-zero boundaries, validation of `Inf`/out-of-range inputs, tolerance
semantics, convergence order, the proxy-secret gate, and the `/demo/solve`
caps.
