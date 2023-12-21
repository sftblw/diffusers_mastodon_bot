from dataclasses import dataclass
from typing import *

from diffusers_mastodon_bot.diffusion.custom_token_helper import CustomTokenHelper


@dataclass
class PipeInfo:
    pipe: Any
    pipe_kwargs_info: Dict[str, Any]
    custom_token_helper: CustomTokenHelper
