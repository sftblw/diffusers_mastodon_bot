import logging

from diffusers_mastodon_bot.locales.locale_data import LocaleData

logger = logging.getLogger(__name__)


class LocaleRes:
    def __init__(self, locale: str = 'en-US'):
        self.diffusion_game = LocaleData(
            locale, ["./locales/diffusion_game.$locale.ftl"]
        )
