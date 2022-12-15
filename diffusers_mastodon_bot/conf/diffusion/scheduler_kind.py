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
