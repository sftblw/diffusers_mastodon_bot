import logging
from typing import *
from diffusers.utils.import_utils import is_xformers_available

import torch

from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion \
    import StableDiffusionLongPromptWeightingPipeline as StableDiffusionLpw


logger = logging.getLogger(__name__)


def create_diffusers_pipeline(device_name='cuda', pipe_kwargs: Optional[Dict[str, Any]] = None):
    if pipe_kwargs is None:
        pipe_kwargs = {}

    pipe_kwargs = pipe_kwargs.copy()

    kwargs_defaults = {
        "pretrained_model_name_or_path": 'hakurei/waifu-diffusion',
        'revision': 'fp16'
    }

    for key, value in kwargs_defaults.items():
        if key not in pipe_kwargs:
            pipe_kwargs[key] = value

    model_name_or_path = pipe_kwargs['pretrained_model_name_or_path']
    del pipe_kwargs['pretrained_model_name_or_path']

    torch_dtype = torch.float32
    if 'torch_dtype' in pipe_kwargs:
        dtype_param = pipe_kwargs['torch_dtype']
        del pipe_kwargs['torch_dtype']

        if dtype_param == 'torch.float16':
            torch_dtype = torch.float16

    if 'scheduler' in pipe_kwargs:
        scheduler_param = pipe_kwargs['scheduler']
        del pipe_kwargs['scheduler']

        if scheduler_param == 'euler':
            from diffusers import EulerDiscreteScheduler
            pipe_kwargs['scheduler'] = EulerDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        elif scheduler_param == 'euler_a':
            from diffusers import EulerAncestralDiscreteScheduler
            pipe_kwargs['scheduler'] = EulerAncestralDiscreteScheduler.from_pretrained(model_name_or_path,
                                                                                       subfolder="scheduler")
        elif scheduler_param == 'dpm_solver++':
            from diffusers import DPMSolverMultistepScheduler
            pipe_kwargs['scheduler'] = DPMSolverMultistepScheduler.from_pretrained(model_name_or_path,
                                                                                   subfolder="scheduler")

    pipe: StableDiffusionLpw = StableDiffusionLpw.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        **pipe_kwargs
    )

    pipe = pipe.to(device_name)
    pipe.enable_attention_slicing()

    if is_xformers_available():
        try:
            pipe.unet.enable_xformers_memory_efficient_attention(True)
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    pipe_kwargs['pretrained_model_name_or_path'] = model_name_or_path
    pipe_kwargs['torch_dtype'] = 'torch.float16' if torch_dtype == torch.float16 else 'torch.float32'
    pipe_kwargs['scheduler'] = str(type(pipe.scheduler).__name__)

    return pipe, pipe_kwargs