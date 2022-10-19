from typing import *


class Prompts(TypedDict):
    positive: str
    negative: str
    negative_with_default: Optional[str]


class ProcArgsContext:
    def __init__(self,
                 prompts: Prompts,
                 proc_kwargs: Dict[str, Any],
                 target_image_count: int,
                 ):
        self.prompts = prompts
        self.proc_kwargs = proc_kwargs
        self.target_image_count = target_image_count
