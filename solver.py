"""2D Poisson solver: -∇²T = f(x,y), T=g on the Dirichlet boundary.

Discretization: 5-point finite differences, O(h²).
Methods: direct DST/FFT (valid for ANY source term), PCG, SOR with analytical
optimal ω, Jacobi, Gauss–Seidel.
References: Demmel CS267, Yang & Gobbert 2009, MIT 18.336, Higham 2002,
Nocedal & Wright.
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import dst

from models import PoissonRequest

# Hard wall-clock budget for iterative methods. The direct DST path is O(N log N)
# and never needs this, but Jacobi/GS can otherwise run for minutes at n=255.
TIME_LIMIT_S = float(os.getenv("POISSON_TIME_LIMIT_S", "10"))


class _TimeLimit(Exception):
    """Raised inside a PCG callback to abort a request that exceeds the budget."""


def omega_opt(n: int) -> float:
    """Optimal SOR ω for 2D Poisson with Dirichlet BC: 2/(1+sin(π/(n+1)))."""
    h = 1.0 / (n + 1)  # interior spacing
    return 2.0 / (1.0 + np.sin(np.pi * h))


def build_grid(req: PoissonRequest):
    n = req.n
    h = req.length / (n + 1)
    x = np.linspace(h, req.length - h, n)  # interior points
    y = np.linspace(h, req.length - h, n)
    X, Y = np.meshgrid(x, y, indexing="xy")  # [j, i]
    xn = X / req.length
    yn = Y / req.length
    if req.source_mode == "constant":
        f = np.full_like(X, req.source_amplitude)
    elif req.source_mode == "gaussian":
        dx = (xn - req.heat_source_x) ** 2
        dy = (yn - req.heat_source_y) ** 2
        sigma = req.heat_sigma
        f = req.source_amplitude * np.exp(-(dx + dy) / (2 * sigma**2))
    else:  # sine
        f = req.source_amplitude * np.sin(np.pi * xn) * np.sin(np.pi * yn)
    return X, Y, f, h


def build_system(n: int, h: float):
    """5-point discrete negative Laplacian. A is symmetric positive definite."""
    if n == 1:
        return sp.csr_matrix(np.array([[4.0 / (h * h)]], dtype=float))
    e = np.ones(n)
    # Shape (n,n) explicit: otherwise sp.diags infers size = len(data)+offset = n+1
    T = sp.diags([-e, 2 * e, -e], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    I = sp.eye(n, format="csr")
    A = (sp.kron(I, T, format="csr") + sp.kron(T, I, format="csr")) / (h * h)
    return A.tocsr()


def rhs_vector(f: np.ndarray, boundary: float, h: float) -> np.ndarray:
    b = f.ravel().copy()
    n = f.shape[0]
    if boundary != 0.0:
        c = boundary / (h * h)
        # First row j=0 (y=h): bottom neighbor is the boundary
        b[0:n] += c
        # Last row j=n-1
        b[(n - 1) * n : n * n] += c
        # Column i=0
        b[0::n] += c
        # Column i=n-1
        b[n - 1 :: n] += c
    return b


def unflatten(v: np.ndarray, n: int) -> np.ndarray:
    return v.reshape((n, n))


def solve_fft(f: np.ndarray, boundary: float, h: float) -> np.ndarray:
    """Fast direct solution via discrete sine transform (DST-I).

    2D Poisson with Dirichlet BC: -∇²T = f.
    The discrete -∇² has eigenvectors sin(kπi/(n+1))·sin(lπj/(n+1)) and eigenvalues
    λ_kl = (2-2cos(kπ/(n+1)))/h² + (2-2cos(lπ/(n+1)))/h².
    DST-I: dst(dst(x)) = 2·(n+1)·x  ⇒  inverse = dst(dst(·))/(2(n+1)) per axis.
    """
    n = f.shape[0]
    F = dst(dst(f, type=1, axis=0), type=1, axis=1)
    k = np.arange(1, n + 1)
    theta = k * np.pi / (n + 1)
    lam1d = (2 - 2 * np.cos(theta)) / (h * h)
    LAM = lam1d[:, None] + lam1d[None, :]
    T_hat = F / LAM
    # Inverse DST-I: both axes, factor 1/(2*(n+1)) each
    T = dst(dst(T_hat, type=1, axis=0), type=1, axis=1) / (4.0 * (n + 1) ** 2)
    return T + boundary


def _neighbor_sum(T: np.ndarray, g: float) -> np.ndarray:
    """Sum of the 4 neighbors for each cell; outside the domain the value is g (Dirichlet)."""
    S = np.empty_like(T)
    S[1:-1, 1:-1] = T[:-2, 1:-1] + T[2:, 1:-1] + T[1:-1, :-2] + T[1:-1, 2:]
    S[0, 1:-1] = g + T[1, 1:-1] + T[0, :-2] + T[0, 2:]
    S[-1, 1:-1] = T[-2, 1:-1] + g + T[-1, :-2] + T[-1, 2:]
    S[1:-1, 0] = g + T[1:-1, 1] + T[:-2, 0] + T[2:, 0]
    S[1:-1, -1] = T[1:-1, -2] + g + T[:-2, -1] + T[2:, -1]
    S[0, 0] = 2 * g + T[1, 0] + T[0, 1]
    S[0, -1] = 2 * g + T[1, -1] + T[0, -2]
    S[-1, 0] = 2 * g + T[-2, 0] + T[-1, 1]
    S[-1, -1] = 2 * g + T[-2, -1] + T[-1, -2]
    return S


def laplacian_grid(T: np.ndarray, boundary: float, h: float) -> np.ndarray:
    """Return A @ T.ravel() for the 5-point Dirichlet operator (matrix-free).

    ``A`` only couples interior unknowns; the boundary values live in ``b``
    (via ``rhs_vector``). Out-of-domain neighbours therefore contribute zero
    here — using ``boundary`` as the outside value would build the *full*
    operator and make the residual wrong whenever ``boundary != 0``.
    """
    del boundary  # kept in the signature for a stable, self-documenting API
    return (4.0 * T - _neighbor_sum(T, 0.0)) / (h * h)


def relative_residual(T: np.ndarray, b: np.ndarray, boundary: float, h: float) -> float:
    Ax = laplacian_grid(T, boundary, h).ravel()
    return float(np.linalg.norm(b - Ax) / (np.linalg.norm(b) or 1.0))


def solve_iterative(A, b: np.ndarray, req: PoissonRequest, f_grid: np.ndarray | None = None):
    """Return (x, iteration_count, info). Uses the solver matching the requested method."""
    n = req.n
    tol = req.tolerance
    maxiter = req.max_iter
    norm_b = np.linalg.norm(b) or 1.0

    t_start = time.perf_counter()

    if req.method == "pcg":
        # Jacobi diagonal preconditioner. On this uniform grid the diagonal is
        # constant, so M is a scalar multiple of I and this is mathematically a
        # no-op; kept for API compatibility and for future non-uniform grids.
        try:
            diag_inv = 1.0 / A.diagonal()
            M = spla.LinearOperator(A.shape, matvec=lambda x: diag_inv * x)
        except Exception:
            M = None
        counter = {"n": 0}
        last_x = {"x": None}

        def cb(xk):
            counter["n"] += 1
            last_x["x"] = np.array(xk, copy=True)
            if time.perf_counter() - t_start > TIME_LIMIT_S:
                raise _TimeLimit

        try:
            try:
                x, info = spla.cg(A, b, rtol=tol, atol=0, maxiter=maxiter, M=M, callback=cb)
            except TypeError:
                counter["n"] = 0
                last_x["x"] = None
                x, info = spla.cg(A, b, tol=tol, maxiter=maxiter, M=M, callback=cb)
        except _TimeLimit:
            x = last_x["x"] if last_x["x"] is not None else np.zeros_like(b)
            return x, counter["n"], 2
        return x, counter["n"], info

    if req.method in ("jacobi", "gauss_seidel", "sor"):
        if req.method == "sor":
            omega = omega_opt(n) if req.omega is None else float(req.omega)
        else:
            omega = 1.0
        D = A.diagonal()
        x = np.zeros_like(b)
        T = unflatten(x, n).copy()
        if f_grid is None:
            # Reconstruct f from b without the boundary term: f = b minus (g/h² on edges)
            raise ValueError("solve_iterative requires f_grid (source without boundary)")
        F = f_grid
        h_val = float(np.sqrt(4.0 / D[0]))
        h2 = h_val * h_val
        g = float(req.boundary)
        jj, ii = np.indices((n, n))
        red = (jj + ii) % 2 == 0
        black = ~red
        it = 0
        res = float("inf")
        timed_out = False
        is_red_black = req.method in ("gauss_seidel", "sor")
        for it in range(1, maxiter + 1):
            if is_red_black:
                # Update reds with current blacks, then blacks with already-updated reds
                S = _neighbor_sum(T, g)
                target = (1 - omega) * T + omega * 0.25 * (S + h2 * F)
                T[red] = target[red]
                S = _neighbor_sum(T, g)
                target = (1 - omega) * T + omega * 0.25 * (S + h2 * F)
                T[black] = target[black]
            else:
                S = _neighbor_sum(T, g)
                T = (1 - omega) * T + omega * 0.25 * (S + h2 * F)
            if it % 5 == 0 or it == maxiter:
                r = b - (A @ T.ravel())
                res = float(np.linalg.norm(r) / norm_b)
                if res < tol:
                    break
                if time.perf_counter() - t_start > TIME_LIMIT_S:
                    timed_out = True
                    break
        x = T.ravel()
        r = b - (A @ x)
        res = float(np.linalg.norm(r) / norm_b)
        info = 2 if timed_out else (0 if res < tol else 1)
        return x, it, info

    raise ValueError(f"Unsupported method: {req.method}")


def solve_poisson(req: PoissonRequest) -> dict:
    t0 = time.perf_counter()
    X, Y, f, h = build_grid(req)
    n = req.n
    b = rhs_vector(f, req.boundary, h)

    if req.method == "fft":
        # Direct DST solve: exact for the discrete system, any source term.
        T = solve_fft(f, req.boundary, h)
        it = 0
        info = 0
        # Matrix-free residual: avoid building A just to measure the residual.
        res = relative_residual(T, b, req.boundary, h)
    else:
        A = build_system(n, h)
        x, it, info = solve_iterative(A, b, req, f_grid=f)
        T = unflatten(x, n)
        res = float(np.linalg.norm(b - A @ x) / (np.linalg.norm(b) or 1.0))

    if not np.all(np.isfinite(T)):
        raise RuntimeError("non-finite solution (check input magnitudes)")

    finite = bool(np.isfinite(res))
    # 1 % numerical slack: iterative stopping rules can land just above rtol.
    success = bool(finite and res <= req.tolerance * 1.01)
    error = None
    if not success:
        if not finite:
            error = "non-finite residual"
        elif info == 2:
            error = f"time limit of {TIME_LIMIT_S:.0f}s exceeded before reaching tolerance"
        elif info == 1:
            error = "maximum iterations reached before reaching tolerance"
        else:
            error = f"residual {res:.3e} is above requested tolerance {req.tolerance:.1e}"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "success": success,
        "method": req.method,
        "direct": req.method == "fft",
        "grid_n": n,
        "iterations": int(it),
        "residual": res,
        "solve_time_ms": elapsed_ms,
        "temp_min": float(np.min(T)),
        "temp_max": float(np.max(T)),
        "temp_mean": float(np.mean(T)),
        "field": T.tolist(),
        "error": error,
    }
