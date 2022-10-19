import abc
import io
import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import *

import diffusers.pipelines
import transformers
from PIL.Image import Image
from torch import autocast

from .bot_request_handler import BotRequestHandler
from .bot_request_context import BotRequestContext
from .diffusion_runner import DiffusionRunner
from .proc_args_context import ProcArgsContext
from ..utils import image_grid


class DiffuseMeHandler(BotRequestHandler):
    def __init__(self,
                 pipe: diffusers.pipelines.StableDiffusionPipeline,
                 tag_name: str = 'diffuse_me',
                 allow_self_request_only: bool = False
                 ):
        self.pipe = pipe
        self.tag_name = tag_name
        self.allow_self_request_only = allow_self_request_only
        self.re_strip_special_token = re.compile('<\|.*?\|>')

    def is_eligible_for(self, ctx: BotRequestContext) -> bool:
        contains_hash = ctx.contains_tag_name(self.tag_name)
        if not contains_hash:
            return False

        return (
            ( ctx.mentions_bot() and ctx.not_from_self() and not self.allow_self_request_only)
            or
            not ctx.not_from_self()
        )

    def respond_to(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        # start
        in_progress_status = self.reply_in_progress(ctx, args_ctx)

        diffusion_result: DiffusionRunner.Result = DiffusionRunner.run_diffusion_and_upload(self.pipe, ctx, args_ctx)
        reply_message = "\ntime: " + diffusion_result['time_took']

        logging.info(f'building reply text')

        def detect_args_and_print(args_name):
            if args_ctx.proc_kwargs is not None and args_name in args_ctx.proc_kwargs:
                return '\n' + f'{args_name}: {args_ctx.proc_kwargs[args_name]}'
            else:
                return ''

        reply_message += detect_args_and_print('num_inference_steps')
        reply_message += detect_args_and_print('guidance_scale')

        if diffusion_result["has_any_nsfw"]:
            reply_message += '\n\n' + 'nsfw content detected, some of result will be a empty image'

        reply_message += '\n\n' + f'prompt: \n{args_ctx.prompts["positive"]}'

        if args_ctx.prompts['negative'] is not None:
            reply_message += '\n\n' + f'negative prompt (without default): \n{args_ctx.prompts["negative"]}'

        if len(reply_message) >= 450:
            reply_message = reply_message[0:400] + '...'

        media_ids = [image_posted['id'] for image_posted in diffusion_result["images_list_posted"]]
        if diffusion_result["has_any_nsfw"] and ctx.bot_ctx.no_image_on_any_nsfw:
            media_ids = None

        spoiler_text = '[done] ' + args_ctx.prompts['positive'][0:20] + '...'

        reply_target_status = ctx.status if ctx.bot_ctx.delete_processing_message else in_progress_status
        ctx.mastodon.status_reply(reply_target_status, reply_message,
                                  media_ids=media_ids,
                                  visibility=ctx.reply_visibility,
                                  spoiler_text=spoiler_text,
                                  sensitive=True
                                  )

        if ctx.bot_ctx.delete_processing_message:
            ctx.mastodon.status_delete(in_progress_status)

        logging.info(f'sent')

        return True

    def reply_in_progress(self, ctx: BotRequestContext, args_ctx: ProcArgsContext):
        processing_body = DiffusionRunner.make_processing_body(self.pipe, args_ctx)
        in_progress_status = ctx.reply_to(status=ctx.status,
                                          body=processing_body if len(processing_body) > 0 else 'processing...',
                                          spoiler_text='processing...' if len(processing_body) > 0 else None,
                                          )
        return in_progress_status
