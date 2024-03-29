from dataclasses import dataclass, field

from diffusers_mastodon_bot.conf.app.behavior_conf import BehaviorConf
from diffusers_mastodon_bot.conf.app.image_gen_conf import ImageGenConf


@dataclass
class AppConf:
    locale: str = 'en-US'
    behavior: BehaviorConf = field(default_factory=BehaviorConf)
    image_gen: ImageGenConf = field(default_factory=ImageGenConf)
