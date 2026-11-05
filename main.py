"""2D Poisson Solver API — steady-state heat conduction, electrostatics, potential flow.

Solves -∇²T = f(x,y), with T = boundary on the entire boundary (Dirichlet).
This is a stationary numerical solver. It does not model transient phenomena,
Navier-Stokes, or general CFD.
Default method: direct DST/FFT, valid for ANY source term.
Also: PCG, SOR (analytical optimal ω), Jacobi, Gauss-Seidel.
"""

from __future__ import annotations

import logging
import math
import os
import time

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from models import PoissonRequest, PoissonResponse
from solver import TIME_LIMIT_S, solve_poisson

logger = logging.getLogger(__name__)

app = FastAPI(
    title="2D Poisson Solver API",
    description=(
        "Stationary 2D Poisson solver: -∇²T = f(x,y) on a square domain with "
        "Dirichlet boundaries. Intended for steady-state heat conduction, "
        "electrostatics (potential), and potential flow. This API does not "
        "model transient or Navier-Stokes/CFD problems. "
        "Default: direct DST/FFT solver for any source term. "
        "Also PCG, SOR, Jacobi, Gauss-Seidel."
    ),
    version="1.2.0",
)

START = time.time()
PORT = int(os.getenv("PORT", "8000"))


def _json_safe(obj):
    """Replace non-finite floats so JSONResponse can serialize validation errors.

    Pydantic rejects Inf/NaN correctly, but its error payload echoes the
    offending input (``inf``). Starlette's JSONResponse uses ``allow_nan=False``
    and would turn that into a 500. We keep the 422 and show "inf"/"nan".
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


def verify_rapidapi_secret(
    x_rapidapi_proxy_secret: str | None = Header(
        default=None, alias="X-RapidAPI-Proxy-Secret"
    ),
):
    """Reject direct traffic when RAPIDAPI_PROXY_SECRET is configured.

    RapidAPI adds this header to every proxied request. If the env var is
    unset (local development), the check is disabled.
    """
    expected = os.getenv("RAPIDAPI_PROXY_SECRET")
    if not expected:
        return
    if x_rapidapi_proxy_secret != expected:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing X-RapidAPI-Proxy-Secret",
        )


def _solve_impl(preq: PoissonRequest) -> PoissonResponse:
    try:
        result = solve_poisson(preq)
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 — numerical error → 422, not an ugly 500
        logger.exception("Unexpected solver error")
        raise HTTPException(status_code=422, detail=f"Numerical error: {e}") from e
    return PoissonResponse(**result)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START, 1),
        "service": "poisson-solver-api",
        "port": PORT,
    }


@app.get("/methods")
def methods_info():
    return {
        "default_method": "fft",
        "methods": ["fft", "pcg", "sor", "jacobi", "gauss_seidel"],
        "time_limit_s": TIME_LIMIT_S,
        "notes": {
            "fft": (
                "Direct DST solve, O(N log N). Valid for ANY source_mode "
                "(constant, gaussian, sine). Exact for the discrete system. "
                "Recommended default."
            ),
            "pcg": (
                "Conjugate Gradient with a Jacobi preconditioner. On this uniform "
                "grid Jacobi is a no-op; use fft unless you need an iterative method."
            ),
            "sor": (
                "SOR with analytical optimal omega 2/(1+sin(pi/(n+1))) "
                "when omega is omitted."
            ),
            "jacobi": "Classical Jacobi, O(N^2) iterations. Comparison / small grids only.",
            "gauss_seidel": (
                "Red-black Gauss-Seidel, ~2x faster than Jacobi. Comparison only."
            ),
        },
        "sources": [
            "Yang & Gobbert 2009 (optimal SOR omega)",
            "MIT 18.336 (DST for Dirichlet Poisson)",
            "Higham 2002 (residual vs error)",
            "Demmel CS267",
            "Nocedal & Wright",
        ],
    }


@app.post(
    "/solve",
    response_model=PoissonResponse,
    dependencies=[Depends(verify_rapidapi_secret)],
)
def solve(preq: PoissonRequest | None = Body(default=None)):
    if preq is None:
        raise HTTPException(status_code=422, detail="Empty or invalid request body")
    return _solve_impl(preq)


@app.post("/demo/solve", response_model=PoissonResponse)
def demo_solve(preq: PoissonRequest | None = Body(default=None)):
    """Public, tightly-limited endpoint used only by the /demo page.

    It exists so the HTML demo keeps working when /solve is protected by
    RAPIDAPI_PROXY_SECRET. The demo is capped to n<=64 and 2000 iterations.
    """
    if preq is None:
        raise HTTPException(status_code=422, detail="Empty or invalid request body")
    preq = preq.model_copy(
        update={
            "n": min(preq.n, 64),
            "max_iter": min(preq.max_iter, 2000),
            "tolerance": max(preq.tolerance, 1e-6),
        }
    )
    return _solve_impl(preq)


@app.get("/demo", response_class=HTMLResponse)
def demo():
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
    <title>2D Poisson Solver API</title>
    <style>
    body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px}
    canvas{border:1px solid #ccc;background:#111}
    .row{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}
    label{display:block;font-size:13px;margin-bottom:4px}
    input,select{padding:6px;border:1px solid #bbb;border-radius:6px}
    button{padding:8px 14px;background:#2563eb;color:#fff;border:0;border-radius:8px;cursor:pointer}
    pre{background:#0b1220;color:#dbe7ff;padding:12px;border-radius:8px;overflow:auto}
    </style></head><body>
    <h1>2D Poisson Solver API</h1>
    <p>Solves <b>-∇²T = f(x,y)</b> with Dirichlet boundary. The default
    <b>FFT/DST</b> method works with <i>any</i> source term and is direct
    (no iterations).</p>
    <div class="row">
      <div><label>Grid n (3–64 in demo)</label><input id="n" type="number" value="64" min="3" max="64"></div>
      <div><label>Source</label><select id="src"><option value="gaussian">gaussian</option><option value="sine">sine</option><option value="constant">constant</option></select></div>
      <div><label>Amplitude</label><input id="amp" type="number" value="100"></div>
      <div><label>Method</label><select id="m"><option value="fft">fft (direct, any source)</option><option value="pcg">pcg</option><option value="sor">sor</option><option value="jacobi">jacobi</option><option value="gauss_seidel">gauss_seidel</option></select></div>
    </div>
    <button onclick="run()">Solve</button>
    <canvas id="cv" width="640" height="640"></canvas>
    <pre id="out">...</pre>
    <script>
    async function run(){
      const n=+document.getElementById('n').value;
      const payload={n, source_mode:document.getElementById('src').value, source_amplitude:+document.getElementById('amp').value, method:document.getElementById('m').value};
      const r=await fetch('/demo/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const j=await r.json();
      document.getElementById('out').textContent=JSON.stringify({success:j.success,method:j.method,direct:j.direct,grid_n:j.grid_n,iter:j.iterations,res:j.residual,ms:j.solve_time_ms,min:j.temp_min,max:j.temp_max,mean:j.temp_mean,error:j.error},null,2);
      const c=document.getElementById('cv'),ctx=c.getContext('2d');
      const f=j.field,mn=j.temp_min,mx=j.temp_max,N=j.grid_n;
      const im=ctx.createImageData(N,N);
      for(let y=0;y<N;y++)for(let x=0;x<N;x++){
        const v=(f[y][x]-mn)/((mx-mn)||1);
        const i=(y*N+x)*4;
        im.data[i]=Math.floor(v*255);im.data[i+1]=Math.floor((1-Math.abs(v-0.5)*2)*80+v*120);im.data[i+2]=Math.floor((1-v)*255);
        im.data[i+3]=255;
      }
      const tmp=document.createElement('canvas');tmp.width=tmp.height=N;tmp.getContext('2d').putImageData(im,0,0);
      ctx.imageSmoothingEnabled=true;ctx.drawImage(tmp,0,0,c.width,c.height);
    }
    run();
    </script></body></html>"""
    return HTMLResponse(html)
