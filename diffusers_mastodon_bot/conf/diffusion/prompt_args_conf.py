from dataclasses import dataclass


@dataclass
class PromptArgsConf:
    allow_ignore_default_negative_prompt: bool = False
