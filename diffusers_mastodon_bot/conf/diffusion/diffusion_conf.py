from dataclasses import dataclass

from diffusers_mastodon_bot.conf.diffusion.pipeline_conf import PipelineConf
from diffusers_mastodon_bot.conf.diffusion.process_conf import ProcessConf
from diffusers_mastodon_bot.conf.diffusion.prompt_args_conf import PromptArgsConf
from diffusers_mastodon_bot.conf.diffusion.prompt_conf import PromptConf


@dataclass
class DiffusionConf:
    device: str = 'cuda'
    pipeline: PipelineConf = PipelineConf()
    process: ProcessConf = ProcessConf()
    prompt: PromptConf = PromptConf()
    prompt_args: PromptArgsConf = PromptArgsConf()
