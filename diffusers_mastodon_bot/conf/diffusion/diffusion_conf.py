from dataclasses import dataclass, field

from diffusers_mastodon_bot.conf.diffusion.embeddings_conf import EmbeddingsConf
from diffusers_mastodon_bot.conf.diffusion.pipeline_conf import PipelineConf
from diffusers_mastodon_bot.conf.diffusion.process_conf import ProcessConf
from diffusers_mastodon_bot.conf.diffusion.prompt_args_conf import PromptArgsConf
from diffusers_mastodon_bot.conf.diffusion.prompt_conf import PromptConf


@dataclass
class DiffusionConf:
    device: str = 'cuda'

    pipeline: PipelineConf = field(default_factory=PipelineConf)
    process: ProcessConf = field(default_factory=ProcessConf)
    prompt: PromptConf = field(default_factory=PromptConf)
    prompt_args: PromptArgsConf = field(default_factory=PromptArgsConf)

    embeddings: EmbeddingsConf = field(default_factory=EmbeddingsConf)
