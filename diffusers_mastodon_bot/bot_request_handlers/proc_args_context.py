from typing import *

from dataclasses import dataclass


class Prompts(TypedDict):
    positive: str
    positive_with_default: str
    negative: Optional[str]
    negative_with_default: Optional[str]


# https://docs.python.org/3.7/library/dataclasses.html
@dataclass
class ProcArgsContext:
    prompts: Prompts
    proc_kwargs: Dict[str, Any]
    target_image_count: int
    pipe_kwargs: Dict[str, Any]
