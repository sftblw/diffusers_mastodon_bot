from dataclasses import dataclass
from typing import Optional

from diffusers_mastodon_bot.conf.diffusion.scheduler_kind import SchedulerKind


@dataclass
class PipelineConf:
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    torch_dtype: str = 'torch.float16'
    scheduler: SchedulerKind = SchedulerKind.DPM_SOLVER_PP
    device_name: str = 'cuda'
    use_safety_checker: bool = True
