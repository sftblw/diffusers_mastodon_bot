from dataclasses import dataclass
from enum import Enum
from typing import Optional

import diffusers.schedulers as schedulers


class SchedulerConf(Enum):
    DDIM = schedulers.DDIMScheduler
    DDPM = schedulers.DDPMScheduler
    IPND = schedulers.IPNDMScheduler
    EULER_A = schedulers.EulerAncestralDiscreteScheduler
    EULER = schedulers.EulerDiscreteScheduler
    DPM_SOLVER = schedulers.DPMSolverSinglestepScheduler
    DPM_SOLVER_PP = schedulers.DPMSolverMultistepScheduler


@dataclass
class ProcessConf:
    width: int = 512
    height: int = 512
    num_inference_steps: int = 28
    guidance_scale: float = 8.0
    strength: float = 0.55



@dataclass
class PipelineConf:
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    torch_dtype: str = 'torch.float16'
    scheduler: SchedulerConf = SchedulerConf.DPM_SOLVER_PP
    device_name: str = 'cuda'
    use_safety_checker: bool = True


@dataclass
class DiffusionConf:
    pipeline: PipelineConf
    process: ProcessConf
