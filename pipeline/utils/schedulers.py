"""
Shared scheduler factory dict for SDXL-based providers.
FLUX and other non-SDXL providers ignore these.
"""
try:
    from diffusers import (
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DDIMScheduler,
        EulerAncestralDiscreteScheduler,
    )
except (ImportError, RuntimeError):
    EulerDiscreteScheduler = None            # type: ignore[assignment]
    DPMSolverMultistepScheduler = None       # type: ignore[assignment]
    DDIMScheduler = None                     # type: ignore[assignment]
    EulerAncestralDiscreteScheduler = None   # type: ignore[assignment]

SCHEDULERS: dict[str, object] = {
    "euler":          (lambda cfg: EulerDiscreteScheduler.from_config(cfg)),
    "dpm++2m":        (lambda cfg: DPMSolverMultistepScheduler.from_config(cfg)),
    "dpm++2m_karras": (lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True)),
    "euler_a":        (lambda cfg: EulerAncestralDiscreteScheduler.from_config(cfg)),
    "ddim":           (lambda cfg: DDIMScheduler.from_config(cfg)),
}
