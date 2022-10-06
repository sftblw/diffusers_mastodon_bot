import logging
import sys
from pathlib import Path
from typing import *
import json

from mastodon import Mastodon

import torch

from diffusers import StableDiffusionPipeline

from diffusers_mastodon_bot.app_stream_listener import AppStreamListener
from pipelines.stable_diffusion.safety_checker_dummy import StableDiffusionSafetyCheckerDummy


def create_diffusers_pipeline(device_name='cuda'):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
        # safety_checker=StableDiffusionSafetyCheckerDummy()
    )

    pipe = pipe.to(device_name)
    pipe.enable_attention_slicing()
    return pipe


def create_diffusers_pipeline_cpu(device_name='cpu'):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)

    pipe = pipe.to(device_name)
    return pipe


def read_text_file(filename: str) -> Union[str, None]:
    path = Path(filename)
    if not Path(filename).is_file():
        return None

    content = path.read_text(encoding='utf8').strip()
    if len(content) == 0:
        return None

    return content


def load_json_dict(filename: str) -> Union[None, Dict[str, any]]:
    result = read_text_file(filename)
    if result is not None:
        return json.loads(result)
    else:
        return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("diffusers_mastodon_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

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

    proc_kwargs = load_json_dict('./config/proc_kwargs.json')
    app_stream_listener_kwargs = load_json_dict('./config/app_stream_listener_kwargs.json')
    if app_stream_listener_kwargs is None:
        app_stream_listener_kwargs = {}

    logging.info('starting')
    mastodon = Mastodon(
        access_token=access_token,
        api_base_url=endpoint_url
    )

    logging.info('info checking')
    account = mastodon.account_verify_credentials()
    my_url = account['url']
    my_acct = account['acct']
    logging.info(f'you are, acct: {my_acct} / url: {my_url}')

    logging.info('loading model')
    device_name = 'cuda'
    pipe = create_diffusers_pipeline(device_name)
    # pipe = create_diffusers_pipeline_cpu(device_name)

    logging.info('creating listener')
    listener = AppStreamListener(mastodon, pipe,
                                 mention_to_url=my_url, tag_name='diffuse_me',
                                 toot_listen_start=toot_listen_start,
                                 toot_listen_end=toot_listen_end,
                                 device=device_name,
                                 proc_kwargs=proc_kwargs,
                                 **app_stream_listener_kwargs
                                 )

    mastodon.stream_user(listener, run_async=False, timeout=10000)


if __name__ == '__main__':
    main()
