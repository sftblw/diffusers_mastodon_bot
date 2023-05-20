from dataclasses import dataclass, field

from diffusers_mastodon_bot.conf.message.toot_listen_msg_conf import TootListenMsgConf


@dataclass
class MessageConf:
    toot_listen_msg: TootListenMsgConf = field(default_factory=TootListenMsgConf)
