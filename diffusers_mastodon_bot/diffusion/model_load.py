import logging
from pathlib import Path
from typing import *

from diffusers.utils.import_utils import is_xformers_available

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion \
    import StableDiffusionLongPromptWeightingPipeline as StableDiffusionLpw
from diffusers_mastodon_bot.conf.diffusion.embeddings_conf import EmbeddingsConf
from diffusers_mastodon_bot.conf.diffusion.pipeline_conf import PipelineConf
from diffusers_mastodon_bot.diffusion.custom_token_helper import CustomTokenHelper
from diffusers_mastodon_bot.diffusion.pipe_info import PipeInfo

logger = logging.getLogger(__name__)


def _enable_xformers(pipe: StableDiffusionLpw):
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


def _create_diffusers_pipeline(conf: PipelineConf) -> Tuple[StableDiffusionLpw, Dict[str, Any]]:
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


def _load_embeddings(embeddings_path: str, custom_token_helper: CustomTokenHelper):
    embeddings_path: Path = Path(embeddings_path)
    if embeddings_path.is_file():
        raise ValueError(f"embeddings_path {embeddings_path} already exists and it is a file, not a directory.")

    if not embeddings_path.exists():
        embeddings_path.mkdir()

    files: List[Path] = list(embeddings_path.glob("./*.pt"))

    if len(files) == 0:
        return

    for file in files:
        file: Path = file  # guide IDE

        try:
            new_token_name = file.stem

            num_embedding_raw = torch.load(file, map_location='cpu')
            if isinstance(num_embedding_raw, dict):
                if 'name' in num_embedding_raw.keys():
                    new_token_name = str(num_embedding_raw['name']).strip()

                new_embedding_tensor: torch.Tensor = num_embedding_raw['string_to_param']['*']

            elif isinstance(num_embedding_raw, torch.Tensor):
                new_embedding_tensor: torch.Tensor = num_embedding_raw
            else:
                raise ValueError(f"I don't know format of the embedding: {file}, "
                                 f"embedding type is: {type(num_embedding_raw)}")

            # if new_embedding_tensor.shape[0] != 1:
            #     raise NotImplementedError(f"Embedding with two or more length is not supported: {file}")

            assert new_embedding_tensor.shape[1] == 768, "embedding should be in 768 dim"

            # load custom name text file, or not
            name_file = Path(file.stem + '.txt')
            if name_file.exists():
                try:
                    new_token_name = Path(file.stem + '').read_text().split('\n')[0].strip()
                except Exception as ex:
                    logger.log(f"Can't read name text file {name_file}: {ex}")

            # set embedding tensor values
            new_token_name = custom_token_helper.add_new_custom_token(
                new_token_name=new_token_name,
                new_embedding_tensor=new_embedding_tensor
            )

            logging.info(f'loaded embedding: {new_token_name}')

        except Exception as ex:
            logger.warn(f'failed to load embedding {file}: {ex}')

    torch.cuda.empty_cache()

    return custom_token_helper


class ModelLoader:
    """simple wrapper for model loading logics"""

    @staticmethod
    def load(pipe_conf: PipelineConf, embeddings_conf: EmbeddingsConf)\
            -> PipeInfo:

        pipe, pipe_kwargs_info = _create_diffusers_pipeline(pipe_conf)

        _enable_xformers(pipe)

        custom_token_helper = CustomTokenHelper(
            prefix=embeddings_conf.prefix,
            text_encoder=pipe.text_encoder,  # type: ignore
            tokenizer=pipe.tokenizer  # type: ignore
        )

        if embeddings_conf.load_embeddings:
            _load_embeddings(
                embeddings_path=embeddings_conf.embeddings_path,
                custom_token_helper=custom_token_helper  # output
            )

        return PipeInfo(
            pipe, pipe_kwargs_info, custom_token_helper
        )
