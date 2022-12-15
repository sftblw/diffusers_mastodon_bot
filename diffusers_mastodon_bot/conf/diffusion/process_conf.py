from dataclasses import dataclass


@dataclass
class ProcessConf:
    width: int = 512
    height: int = 512
    num_inference_steps: int = 28
    guidance_scale: float = 8.0
    strength: float = 0.55
