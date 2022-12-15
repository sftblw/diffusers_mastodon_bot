from typing import *

from diffusers_mastodon_bot.conf.app.behavior_conf import BehaviorConf
from diffusers_mastodon_bot.conf.app.image_gen_conf import ImageGenConf


class BotContext:
    def __init__(self,
                 bot_acct_url: str,
                 behavior_conf: BehaviorConf,
                 image_gen_conf: ImageGenConf,
                 default_visibility: str,
                 device_name: str
                 ):
        self.bot_acct_url = bot_acct_url

        self.behavior_conf = behavior_conf
        self.image_gen_conf = image_gen_conf

        self.default_visibility = default_visibility
        self.device_name = device_name
