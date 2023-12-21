import logging
from typing import *

import atexit
from pathlib import Path
from enum import Enum

import traceback

import mastodon
from mastodon import Mastodon

from diffusers_mastodon_bot.bot_context import BotContext
from diffusers_mastodon_bot.bot_request_handlers.bot_request_context import BotRequestContext
from diffusers_mastodon_bot.bot_request_handlers.bot_request_handler import BotRequestHandler
from diffusers_mastodon_bot.bot_request_handlers.proc_args_context import ProcArgsContext
from diffusers_mastodon_bot.conf.app.app_conf import AppConf
from diffusers_mastodon_bot.conf.diffusion.diffusion_conf import DiffusionConf
from diffusers_mastodon_bot.conf.message.message_conf import MessageConf
from diffusers_mastodon_bot.locales.locale_res import LocaleRes
from diffusers_mastodon_bot.diffusion.param_parser import ParamParser

logger = logging.getLogger(__name__)


class AppStreamListener(mastodon.StreamListener):
    def __init__(
            self,
            mastodon_client,
            mention_to_url: str,
            req_handlers: List[BotRequestHandler],
            diffusion_conf: DiffusionConf,
            app_conf: AppConf,
            message_conf: MessageConf,
            locale_res: LocaleRes,
            pipe_kwargs_info: Dict[str, Any]
    ):
        self.mastodon: Mastodon = mastodon_client
        self.mention_to_url = mention_to_url

        self.diffusion_conf = diffusion_conf
        self.app_conf = app_conf
        self.message_conf = message_conf
        self.locale_res = locale_res

        self.pipe_kwargs_info = pipe_kwargs_info

        if not Path(self.app_conf.behavior.output_save_path).is_dir():
            Path(self.app_conf.behavior.output_save_path).mkdir(parents=True, exist_ok=True)

        self.bot_ctx = BotContext(
            bot_acct_url=mention_to_url,
            behavior_conf=self.app_conf.behavior,
            image_gen_conf=self.app_conf.image_gen,
            locale_res=self.locale_res,
            default_visibility=app_conf.behavior.default_visibility,
            max_visibility=app_conf.behavior.max_visibility,
            min_visibility=app_conf.behavior.min_visibility,
            device_name=self.diffusion_conf.device
        )

        self.param_parser = ParamParser(
            image_gen_conf=self.app_conf.image_gen,
            prompt_conf=self.diffusion_conf.prompt,
            prompt_args_conf=self.diffusion_conf.prompt_args,
            proc_conf=self.diffusion_conf.process,
        )

        self.req_handlers = req_handlers

        boot_msg_cfg = self.message_conf.toot_listen_msg

        def exit_toot():
            if boot_msg_cfg.toot_on_start_end:
                self.mastodon.status_post(boot_msg_cfg.toot_listen_end, visibility=self.bot_ctx.default_visibility)
            pass

        atexit.register(exit_toot)

        logger.info('listening')
        if boot_msg_cfg.toot_on_start_end:
            self.mastodon.status_post(
                boot_msg_cfg.toot_listen_start,
                spoiler_text=boot_msg_cfg.toot_listen_start_cw,
                visibility=self.bot_ctx.default_visibility
            )

    def on_notification(self, notification):
        if 'status' not in notification:
            logger.info('no status found on notification')
            return

        status = notification['status']

        try:
            result = self.handle_updates(status)
            if result.value >= 500:
                logger.warning(f'response failed for {status["url"]}: {result}')
        except Exception as ex:
            logger.error(f'error on notification respond:\n' + traceback.format_exc())
            pass

    # self response, without notification
    def on_update(self, status: Dict[str, Any]):
        super().on_update(status)

        account = status['account']
        if account['url'] != self.mention_to_url:
            return

        try:
            result = self.handle_updates(status)
            if result.value >= 500:
                logger.warning(f'response failed for {status["url"]}')
        except Exception as ex:
            logger.error(f'error on self status respond:\n' + traceback.format_exc())
            pass

    class HandleUpdateResult(Enum):
        success = 200
        no_eligible = 404
        internal_error = 500

    def handle_updates(self, status: Dict[str, Any]) -> HandleUpdateResult:
        req_ctx = BotRequestContext(
            status=status,
            mastodon=self.mastodon,
            bot_ctx=self.bot_ctx
        )

        for handler in self.req_handlers:
            handler: BotRequestHandler = handler
            if not handler.is_eligible_for(req_ctx):
                continue

            prompts, proc_kwargs, target_image_count = \
                self.param_parser.process_common_params(status['content'])

            args_ctx = ProcArgsContext(
                prompts=prompts,
                proc_kwargs=proc_kwargs,
                target_image_count=target_image_count,
                # for model info save
                pipe_kwargs=self.pipe_kwargs_info
            )

            if handler.respond_to(ctx=req_ctx, args_ctx=args_ctx):
                return AppStreamListener.HandleUpdateResult.success
            else:
                return AppStreamListener.HandleUpdateResult.internal_error

        return AppStreamListener.HandleUpdateResult.no_eligible

    def on_unknown_event(self, name, unknown_event=None):
        logger.info(f'unknown event {name}, {unknown_event}')
