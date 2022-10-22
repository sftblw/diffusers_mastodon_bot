from typing import *

from dataclasses import dataclass

class Prompts(TypedDict):
    positive: str
    negative: str
    negative_with_default: Optional[str]


# https://docs.python.org/3.7/library/dataclasses.html
@dataclass
class ProcArgsContext:
    prompts: Prompts
    proc_kwargs: Dict[str, Any]
    target_image_count: int

    