import logging
import sys
from pathlib import Path
from typing import *
import json
from diffusers.utils.import_utils import is_xformers_available

from mastodon import Mastodon
from omegaconf import OmegaConf

from diffusers_mastodon_bot.app_stream_listener import AppStreamListener
from diffusers_mastodon_bot.bot_request_handlers.bot_request_handler import BotRequestHandler
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_handler import DiffuseGameHandler
from diffusers_mastodon_bot.bot_request_handlers.diffuse_me_handler import DiffuseMeHandler
from diffusers_mastodon_bot.bot_request_handlers.diffuse_it_handler import DiffuseItHandler
from diffusers_mastodon_bot.conf.conf_helper import load_structured_conf_yaml


from diffusers_mastodon_bot.conf.instance_conf import InstanceConf
from diffusers_mastodon_bot.conf.diffusion_conf import DiffusionConf
from diffusers_mastodon_bot.conf.toot_listen_msg_conf import TootListenMsgConf

from diffusers_mastodon_bot.model_load import create_diffusers_pipeline


logger = logging.getLogger(__name__)


def read_text_file(filename: str) -> Union[str, None]:
    path = Path(filename)
    if not Path(filename).is_file():
        return None

    content = path.read_text(encoding='utf8').strip()
    if len(content) == 0:
        return None

    return content


def load_json_dict(filename: str) -> Union[None, Dict[str, Any]]:
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

    instance: InstanceConf = load_structured_conf_yaml(InstanceConf, './config/instance.yaml')  # type: ignore
    toot_listen_msg: TootListenMsgConf = load_structured_conf_yaml(TootListenMsgConf, './config/toot_listen_msg.yaml')  # type: ignore

    diffusion_conf: DiffusionConf = load_structured_conf_yaml(DiffusionConf, './config/diffusion.yaml')  # type: ignore

    pipe_conf = diffusion_conf.pipeline
    proc_kwargs = diffusion_conf.process

    app_stream_listener_kwargs = load_json_dict('./config/app_stream_listener_kwargs.json')
    if app_stream_listener_kwargs is None:
        app_stream_listener_kwargs = {}

    diffusion_game_messages = load_json_dict('./config/diffusion_game_messages.json')

    logger.info('starting')
    mastodon = Mastodon(
        access_token=instance.access_token,
        api_base_url=instance.endpoint_url
    )

    logger.info('info checking')
    account = mastodon.account_verify_credentials()
    my_url = account['url']
    my_acct = account['acct']
    logger.info(f'you are, acct: {my_acct} / url: {my_url}')

    logger.info('loading model')

    pipe, pipe_kwargs = create_diffusers_pipeline(pipe_conf)

    logger.info('creating handlers')

    req_handlers: List[BotRequestHandler] = [
        DiffuseMeHandler(
            pipe=pipe,
            tag_name="diffuse_me",
        ),
        DiffuseItHandler(
            pipe=pipe,
            tag_name='diffuse_it'
        ),
        DiffuseGameHandler(
            pipe=pipe,
            tag_name='diffuse_game',
            messages=diffusion_game_messages,  # type: ignore
            response_duration_sec=60 * 30
        )
    ]  # type: ignore

    logger.info('creating listener')
    listener = AppStreamListener(
        mastodon_client=mastodon,
        diffusers_pipeline=pipe,
        mention_to_url=my_url,
        req_handlers=req_handlers,
        toot_listen_msg=toot_listen_msg,
        device=diffusion_conf.pipeline.device_name,
        proc_kwargs=OmegaConf.to_container(proc_kwargs),
        pipe_kwargs=pipe_kwargs,
        **app_stream_listener_kwargs
    )

    mastodon.stream_user(listener, run_async=False, timeout=10000)


if __name__ == '__main__':
    main()
