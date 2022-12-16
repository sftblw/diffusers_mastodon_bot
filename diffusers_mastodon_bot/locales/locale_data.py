import logging
from typing import List

from fluent_compiler.compiler import compile_messages
from fluent_compiler.resource import FtlResource

from diffusers_mastodon_bot.locales.locale_item import LocaleItem


logger = logging.getLogger(__name__)


class LocaleData:
    def __init__(self, locale: str, file_path_list: List[str]):
        def res_in_locale(locale_str: str):
            return [FtlResource.from_file(file_path.replace("$locale", locale_str)) for file_path in file_path_list]

        self._compiled_message_fallback = compile_messages(
            locale=locale,
            resources=res_in_locale('en-US')
        )

        self._compiled_message = compile_messages(
            locale=locale,
            resources=res_in_locale(locale)
        )

    def __getitem__(self, item) -> LocaleItem:
        if item in self._compiled_message.message_functions.keys():
            return LocaleItem(item, self._compiled_message.message_functions[item])
        elif item in self._compiled_message_fallback.message_functions.keys():
            return LocaleItem(item, self._compiled_message_fallback.message_functions[item])
        else:
            logger.warn(f"no localization key found for {item}")
            return LocaleItem(item, None)
