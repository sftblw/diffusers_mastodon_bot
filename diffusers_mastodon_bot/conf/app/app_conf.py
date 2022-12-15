from dataclasses import dataclass

from diffusers_mastodon_bot.conf.app.behavior_conf import BehaviorConf
from diffusers_mastodon_bot.conf.app.image_gen_conf import ImageGenConf


@dataclass
class AppConf:
    behavior: BehaviorConf = BehaviorConf()
    image_gen: ImageGenConf = ImageGenConf()
