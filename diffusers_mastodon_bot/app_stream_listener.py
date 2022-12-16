import logging
from typing import *

import unicodedata
import atexit
import re
from pathlib import Path
from enum import Enum

import traceback

import diffusers.pipelines
import mastodon
from mastodon import Mastodon
from omegaconf import OmegaConf

from diffusers_mastodon_bot.bot_context import BotContext
from diffusers_mastodon_bot.bot_request_handlers.bot_request_context import BotRequestContext
from diffusers_mastodon_bot.bot_request_handlers.bot_request_handler import BotRequestHandler
from diffusers_mastodon_bot.bot_request_handlers.proc_args_context import ProcArgsContext
from diffusers_mastodon_bot.conf.app.app_conf import AppConf
from diffusers_mastodon_bot.conf.diffusion.diffusion_conf import DiffusionConf
from diffusers_mastodon_bot.conf.message.message_conf import MessageConf
from diffusers_mastodon_bot.locales.locale_res import LocaleRes
from diffusers_mastodon_bot.utils import rip_out_html

logger = logging.getLogger(__name__)


class AppStreamListener(mastodon.StreamListener):
    def __init__(
            self,
            mastodon_client,
            diffusers_pipeline: diffusers.pipelines.StableDiffusionPipeline,
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
        self.diffusers_pipeline: diffusers.pipelines.StableDiffusionPipeline = diffusers_pipeline

        self.diffusion_conf = diffusion_conf
        self.app_conf = app_conf
        self.message_conf = message_conf
        self.locale_res = locale_res

        self.pipe_kwargs_info = pipe_kwargs_info

        self.strippers = [
            re.compile(r'@[a-zA-Z0-9._-]+'),
            re.compile(r'#\w+'),
            re.compile(r'[ \r\n\t]+'),
        ]

        if not Path(self.app_conf.behavior.output_save_path).is_dir():
            Path(self.app_conf.behavior.output_save_path).mkdir(parents=True, exist_ok=True)

        self.bot_ctx = BotContext(
            bot_acct_url=mention_to_url,
            behavior_conf=self.app_conf.behavior,
            image_gen_conf=self.app_conf.image_gen,
            locale_res=self.locale_res,
            default_visibility=app_conf.behavior.default_visibility,
            device_name=self.diffusion_conf.device
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
                self.process_common_params(status['content'])

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

    # TODO: refactor into own class
    def process_common_params(self, content: str):
        image_gen_conf = self.app_conf.image_gen
        prompt_conf = self.diffusion_conf.prompt
        proc_kwargs = OmegaConf.to_container(self.diffusion_conf.process)

        logger.info(f'html : {content}')
        content_txt = rip_out_html(content)
        logger.info(f'text : {content_txt}')
        for stripper in self.strippers:
            content_txt = stripper.sub(' ', content_txt).strip()

        content_txt = unicodedata.normalize('NFC', content_txt)
        logger.info(f'text (strip out) : {content_txt}')

        logger.info('starting')

        proc_kwargs = proc_kwargs if proc_kwargs is not None else {}
        proc_kwargs = proc_kwargs.copy()
        if 'width' not in proc_kwargs: proc_kwargs['width'] = 512
        if 'height' not in proc_kwargs: proc_kwargs['height'] = 512

        # param parsing
        tokens = [tok.strip() for tok in content_txt.split(' ')]
        before_args_name = None
        new_content_txt = ''
        target_image_count = image_gen_conf.image_count
        ignore_default_negative_prompt = False

        for tok in tokens:
            if tok.startswith('args.'):
                args_name = tok[5:]
                if args_name == 'ignore_default_negative_prompt':
                    ignore_default_negative_prompt = True
                else:
                    before_args_name = args_name
                continue

            if before_args_name is not None:
                args_value = tok

                if before_args_name == 'orientation':
                    if (
                            args_value == 'landscape' and proc_kwargs['width'] < proc_kwargs['height']
                            or args_value == 'portrait' and proc_kwargs['width'] > proc_kwargs['height']
                    ):
                        width_backup = proc_kwargs['width']
                        proc_kwargs['width'] = proc_kwargs['height']
                        proc_kwargs['height'] = width_backup
                    if args_value == 'square':
                        proc_kwargs['width'] = min(proc_kwargs['width'], proc_kwargs['height'])
                        proc_kwargs['height'] = min(proc_kwargs['width'], proc_kwargs['height'])

                elif before_args_name == 'image_count':
                    if 1 <= int(args_value) <= image_gen_conf.max_image_count:
                        target_image_count = int(args_value)

                elif before_args_name in ['num_inference_steps']:
                    proc_kwargs[before_args_name] = min(int(args_value), 100)

                elif before_args_name in ['guidance_scale']:
                    proc_kwargs[before_args_name] = min(float(args_value), 100.0)

                elif before_args_name in ['strength']:
                    actual_value = None
                    if args_value.strip() == 'low':
                        actual_value = 0.35
                    elif args_value.strip() == 'medium':
                        actual_value = 0.65
                    elif args_value.strip() == 'high':
                        actual_value = 0.8
                    else:
                        actual_value = max(min(float(args_value), 1.0), 0.0)
                    proc_kwargs[before_args_name] = actual_value

                before_args_name = None
                continue

            new_content_txt += ' ' + tok
        if target_image_count > image_gen_conf.max_image_count:
            target_image_count = image_gen_conf.max_image_count
        content_txt = new_content_txt.strip()
        content_txt_negative = None
        if 'sep.negative' in content_txt:
            content_txt_split = content_txt.split('sep.negative')
            content_txt = content_txt_split[0]
            content_txt_negative = ' '.join(content_txt_split[1:]).strip() if len(content_txt_split) >= 2 else None
        logger.info(f'text (after argparse) : {content_txt}')
        logger.info(f'text negative (after argparse) : {content_txt_negative}')
        content_txt_negative_with_default = content_txt_negative
        if prompt_conf.default_negative_prompt is not None and not ignore_default_negative_prompt:
            content_txt_negative_with_default = (
                content_txt_negative
                if content_txt_negative is not None
                else '' + ' ' + prompt_conf.default_negative_prompt
            ).strip()

        prompts = {
            "positive": content_txt,
            "negative": content_txt_negative,
            "negative_with_default": content_txt_negative_with_default
        }

        return prompts, proc_kwargs, target_image_count

    def on_unknown_event(self, name, unknown_event=None):
        logger.info(f'unknown event {name}, {unknown_event}')
