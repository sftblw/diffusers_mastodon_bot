from diffusers_mastodon_bot.conf.diffusion.scheduler_presets.scheduler_kind import WaifuDiffusionXLSchedulerLoader
from enum import Enum

from diffusers import schedulers as schedulers


class SchedulerKind(Enum):
    DDIM = schedulers.DDIMScheduler
    DDPM = schedulers.DDPMScheduler
    IPND = schedulers.IPNDMScheduler
    EULER_A = schedulers.EulerAncestralDiscreteScheduler
    EULER = schedulers.EulerDiscreteScheduler
    DPM_SOLVER = schedulers.DPMSolverSinglestepScheduler
    DPM_SOLVER_PP = schedulers.DPMSolverMultistepScheduler
    DEIS_MULTISTEP = schedulers.DEISMultistepScheduler  # diffusers v0.12.0
    SDE_DPM_SOLVER_PP = schedulers.DPMSolverSDEScheduler

    # preset
    PRESET_WAIFU_DIFFUSION_XL = WaifuDiffusionXLSchedulerLoader

