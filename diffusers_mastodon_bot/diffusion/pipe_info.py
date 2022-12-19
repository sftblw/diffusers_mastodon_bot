from dataclasses import dataclass
from typing import *

from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion \
    import StableDiffusionLongPromptWeightingPipeline as StableDiffusionLpw
from diffusers_mastodon_bot.diffusion.custom_token_helper import CustomTokenHelper


@dataclass
class PipeInfo:
    pipe: StableDiffusionLpw
    pipe_kwargs_info: Dict[str, Any]
    custom_token_helper: CustomTokenHelper
