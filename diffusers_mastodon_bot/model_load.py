import logging
from diffusers.utils.import_utils import is_xformers_available

import torch

from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion \
    import StableDiffusionLongPromptWeightingPipeline as StableDiffusionLpw
from diffusers_mastodon_bot.conf.diffusion.pipeline_conf import PipelineConf

logger = logging.getLogger(__name__)


def enable_xformers(pipe: StableDiffusionLpw):
    if is_xformers_available():
        try:
            pipe.unet.enable_xformers_memory_efficient_attention(True)
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    else:
        logger.info('xformers is not available. not enabling it. ("xformers" gives performance boost)')


def create_diffusers_pipeline(conf: PipelineConf):
    pipe_kwargs = {
        "pretrained_model_name_or_path": conf.pretrained_model_name_or_path,
        "torch_dtype": torch.float16 if conf.torch_dtype == 'torch.float16' else torch.float32,
        "scheduler": conf.scheduler.value.from_pretrained(conf.pretrained_model_name_or_path, subfolder="scheduler"),
    }

    if conf.use_safety_checker:
        pipe_kwargs['safety_checker'] = None
    if conf.revision is not None:
        pipe_kwargs['revision'] = conf.revision

    pipe: StableDiffusionLpw = StableDiffusionLpw.from_pretrained(
        **pipe_kwargs
    )

    pipe = pipe.to(conf.device_name)
    pipe.enable_attention_slicing()

    pipe_kwargs_info = pipe_kwargs.copy()

    pipe_kwargs_info['torch_dtype'] = conf.torch_dtype
    pipe_kwargs_info['scheduler'] = str(type(pipe.scheduler).__name__)

    return pipe, pipe_kwargs_info
