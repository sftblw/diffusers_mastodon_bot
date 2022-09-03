from pathlib import Path
from typing import *

from mastodon import Mastodon

import torch

from diffusers import StableDiffusionPipeline

from diffusers_mastodon_bot.app_stream_listener import AppStreamListener





def create_diffusers_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True
    )

    pipe = pipe.to("cuda")
    return pipe


def read_text_file(filename: str) -> Union[str, None]:
    path = Path(filename)
    if not Path(filename).is_file():
        return None

    content = path.read_text().strip()
    if len(content) == 0:
        return None


if __name__ == '__main__':
    access_token = read_text_file('./config/access_token.txt')
    endpoint_url = read_text_file('./config/endpoint_url.txt')

    if access_token is None:
        print('mastodon access token is required but not found. check ./config/access_token.txt')
        exit()

    if access_token is None:
        print('mastodon endpoint url is required but not found. check ./config/endpoint_url.txt')
        exit()

    toot_listen_start = read_text_file('./config/toot_listen_start.txt')
    toot_listen_end = read_text_file('./config/toot_listen_end.txt')

    print('starting')
    mastodon = Mastodon(
        access_token=access_token,
        api_base_url=endpoint_url
    )

    print('info checking')
    account = mastodon.account_verify_credentials()
    my_acct = mastodon.account_verify_credentials()['acct']
    print(f'you are {my_acct}')

    print('loading model')
    pipe = create_diffusers_pipeline()

    print('creating listener')
    listener = AppStreamListener(mastodon, pipe,
                                 mention_to=my_acct, tag_name='diffuse_me',
                                 toot_listen_start=toot_listen_start,
                                 toot_listen_end=toot_listen_end
                                 )

    mastodon.stream_user(listener, run_async=False, timeout=10000)

