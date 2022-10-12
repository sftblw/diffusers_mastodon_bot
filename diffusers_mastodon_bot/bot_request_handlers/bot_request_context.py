from mastodon import Mastodon

from diffusers_mastodon_bot.bot_context import BotContext
from typing import *

class BotRequestContext:
    def __init__(self,
                 status: Dict[str, any],
                 mastodon: Mastodon,
                 bot_ctx: BotContext,
                 is_self_response: bool,
                 ):
        self.status = status
        self.mastodon: Mastodon = mastodon
        self.bot_ctx = bot_ctx
        self.is_self_response = is_self_response

        self.reply_visibility = status['visibility']
        if self.reply_visibility == 'public' or self.reply_visibility == 'direct':
            self.reply_visibility = 'unlisted'
        self.reply_visibility = 'unlisted'

        # [{'name': 'testasdf', 'url': 'https://don.naru.cafe/tags/testasdf'}]
        self.tag_name_list = set(map(lambda tag: tag['name'], status['tags']))

    def contains_tag_name(self, tag_name):
        return tag_name in self.tag_name_list

    def mentions_bot(self):
        account = self.status['account']
        return account['url'] != self.bot_ctx.bot_acct_url

    def reply_to(self, status: Dict[str, any], body: str, **kwargs):
        if 'visibility' not in kwargs.keys():
            kwargs['visibility'] = self.reply_visibility

        self.mastodon.status_reply(status, body, **kwargs)


