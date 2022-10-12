from typing import Dict


class ProcArgsContext:
    def __init__(self,
                 prompts: str,
                 proc_kwargs: Dict[str, any],
                 target_image_count: int,
                 ):
        self.prompts = prompts
        self.proc_kwargs = proc_kwargs
        self.target_image_count = target_image_count
