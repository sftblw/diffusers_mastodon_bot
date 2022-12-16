from typing import *

from diffusers_mastodon_bot.conf.app.behavior_conf import BehaviorConf
from diffusers_mastodon_bot.conf.app.image_gen_conf import ImageGenConf
from diffusers_mastodon_bot.locales.locale_res import LocaleRes


class BotContext:
    def __init__(self,
                 bot_acct_url: str,
                 behavior_conf: BehaviorConf,
                 image_gen_conf: ImageGenConf,
                 locale_res: LocaleRes,
                 default_visibility: str,
                 device_name: str
                 ):
        self.bot_acct_url = bot_acct_url

        self.behavior_conf = behavior_conf
        self.image_gen_conf = image_gen_conf
        self.locale_res = locale_res

        self.default_visibility = default_visibility
        self.device_name = device_name
