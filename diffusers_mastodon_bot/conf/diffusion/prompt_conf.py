from dataclasses import dataclass


@dataclass
class PromptConf:
    default_negative_prompt: str = \
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit," \
        " fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, " \
        "signature, watermark, username, blurry"
