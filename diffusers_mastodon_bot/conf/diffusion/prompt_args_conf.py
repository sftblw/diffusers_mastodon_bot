from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptArgsConf:
    allow_ignore_default_positive_prompt: Optional[bool] = True
    allow_ignore_default_negative_prompt: Optional[bool] = False
