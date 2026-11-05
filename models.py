from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PoissonRequest(BaseModel):
    """Validated request for the 2D Poisson solver.

    All float fields reject NaN/Inf. The bounds below keep every request
    finite and bounded so a client cannot trigger overflow/NaN loops.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    n: int = Field(default=64, ge=3, le=256, description="Number of grid nodes per side (NxN)")
    length: float = Field(default=1.0, ge=1e-6, le=1e6, description="Domain side length [m]")
    boundary: float = Field(
        default=0.0, ge=-1e6, le=1e6, description="Dirichlet value on all boundaries"
    )
    source_mode: Literal["constant", "gaussian", "sine"] = "gaussian"
    source_amplitude: float = Field(
        default=100.0, ge=-1e6, le=1e6, description="Source amplitude f(x,y)"
    )
    method: Literal["fft", "pcg", "sor", "jacobi", "gauss_seidel"] = Field(
        default="fft",
        description="fft = direct DST solve (any source, recommended default)",
    )
    omega: float | None = Field(
        default=None,
        ge=0.1,
        le=1.99,
        description="SOR relaxation factor; null/omitted = analytical optimal omega",
    )
    tolerance: float = Field(
        default=1e-8,
        ge=1e-10,
        le=1e-2,
        description="Relative residual tolerance (1e-10 is the numerical floor)",
    )
    max_iter: int = Field(default=10000, ge=1, le=20000)
    heat_source_x: float = Field(default=0.5, ge=0, le=1)
    heat_source_y: float = Field(default=0.5, ge=0, le=1)
    heat_sigma: float = Field(default=0.08, ge=1e-3, le=1e6)


class PoissonResponse(BaseModel):
    success: bool
    method: str
    direct: bool = False
    grid_n: int
    iterations: int
    residual: float
    solve_time_ms: float
    temp_min: float
    temp_max: float
    temp_mean: float
    field: list[list[float]] = Field(
        default_factory=list,
        description="Temperature field T(x,y), indexed as field[j][i] (j=y, i=x)",
    )
    error: str | None = None
