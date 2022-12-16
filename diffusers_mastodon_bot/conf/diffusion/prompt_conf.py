from dataclasses import dataclass, field
from typing import *


@dataclass
class PromptConf:
    default_negative_prompt: str = \
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit," \
        " fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, " \
        "signature, watermark, username, blurry"

    filter_positive_regex: List[str] = field(default_factory=lambda: [
        'nsfw', 'sex', 'penis', 'vagina', 'pussy', 'lewd', 'hentai', r'(^|\b)erect\W*?nipples?($|\b)',
        '(^|\b)vore($|\b)', 'bukk?ake', 'no pantsu?'
    ])

    filter_negative_regex: List[str] = field(default_factory=lambda: [

    ])

    replace_positive_regex: List[str] = field(default_factory=lambda: [
        'topless -> bare shoulders'
    ])
