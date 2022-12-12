from dataclasses import dataclass


@dataclass
class TootListenMsgConf:
    toot_listen_start: str = """
#bot #bot_message
diffusers_mastodon_bot starts!
""".strip()
    toot_listen_start_cw: str = "diffusers_mastodon_bot start (cw message)"
    toot_listen_end: str = """
#bot #bot_message
diffusers_mastodon_bot ends! don't send me anymore!
""".strip()