"""Reproducible FFT/DST vs iterative benchmark for the 2D Poisson Solver API.

Run from the repository root:
    python benchmarks/benchmark.py

Times are wall-clock milliseconds for an in-process solve (no HTTP overhead).
"""

from __future__ import annotations

import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import PoissonRequest  # noqa: E402
from solver import solve_poisson  # noqa: E402


def run(n: int, method: str, tolerance: float = 1e-8) -> tuple[float, int, float, bool]:
    req = PoissonRequest(
        n=n,
        method=method,
        source_mode="gaussian",
        tolerance=tolerance,
        max_iter=20000,
    )
    t0 = time.perf_counter()
    result = solve_poisson(req)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return wall_ms, result["iterations"], result["residual"], result["success"]


def main() -> None:
    print(f"{'n':>5}  {'method':>5}  {'wall ms':>9}  {'iters':>7}  {'residual':>10}  ok")
    print("-" * 52)
    for n in (63, 127, 255):
        for method in ("fft", "pcg"):
            # Warm-up, then median of three runs to reduce OS noise.
            run(n, method)
            samples = [run(n, method) for _ in range(3)]
            wall_ms = statistics.median(s[0] for s in samples)
            iters = samples[0][1]
            residual = samples[0][2]
            ok = all(s[3] for s in samples)
            print(
                f"{n:>5}  {method:>5}  {wall_ms:>9.2f}  {iters:>7d}  "
                f"{residual:>10.2e}  {'yes' if ok else 'no'}"
            )


if __name__ == "__main__":
    main()
