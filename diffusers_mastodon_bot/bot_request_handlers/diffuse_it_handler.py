import abc
import io
import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path
import traceback
from typing import *


import diffusers.pipelines
import torch
import transformers
import PIL
from PIL import Image
from torch import autocast

from .bot_request_handler import BotRequestHandler
from .bot_request_context import BotRequestContext
from .diffusion_runner import DiffusionRunner
from .proc_args_context import ProcArgsContext
from ..utils import image_grid


import requests
import glob
from io import BytesIO


def convert_image(image) -> PIL.Image.Image:
    # https://stackoverflow.com/a/9459208/4394750
    background = PIL.Image.new("RGB", image.size, (255, 255, 255))  # type: ignore
    image_split = image.split()
    if len(image_split) >= 4:
        background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
    else:
        background.paste(image)
        image = background.convert("RGB")
    return image.convert("RGB")

def download_image(url) -> Optional[PIL.Image.Image]:
    # this fn is from HF's textual inversion notebook, but modified.
    try:
        response = requests.get(url)
    except:
        return None

    image = PIL.Image.open(BytesIO(response.content))  # type: ignore
    return image


class DiffuseItHandler(BotRequestHandler):
    def __init__(self,
                 pipe: diffusers.pipelines.StableDiffusionImg2ImgPipeline,
                 tag_name: str = 'diffuse_it',
                 allow_self_request_only: bool = False
                 ):
        self.pipe = pipe
        self.tag_name = tag_name
        self.allow_self_request_only = allow_self_request_only
        self.re_strip_special_token = re.compile('<\|.*?\|>')

        self.generator = torch.Generator(device='cuda')

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

        if 'media_attachments' not in ctx.status.keys() or len(ctx.status['media_attachments']) == 0:
            ctx.reply_to(ctx.status, 'no attachment found')
            logging.warning('no attachment found, early returning')
            return True

        attachments: List[Dict[str, Any]] = ctx.status['media_attachments']
        first_attachment = attachments[0]

        if 'url' not in first_attachment:
            ctx.reply_to(ctx.status, 'no url found from attachment')
            logging.warning('no url found from attachment, early returning')
            return True

        try:
            image: Optional[PIL.Image.Image] = download_image(first_attachment['url'])
            if image is None:
                ctx.reply_to(ctx.status, "can't download or convert image")
                logging.warning("can't download or convert image, early returning")
                return True
        except Exception as ex:
            ctx.reply_to(ctx.status, "can't download image")
            logging.warning(f"can't download or convert image: " + "\n  ".join(traceback.format_exception(ex)))

            return True

        # follow orientation of image
        target_width = args_ctx.proc_kwargs['width']
        target_height =  args_ctx.proc_kwargs['height']
        if image.width == image.height and target_width != target_height:
            target_width = min(target_width, target_height)
            target_height = min(target_width, target_height)
        elif ( image.width > image.height and target_width < target_height ) \
            or ( image.width < image.height and target_width > target_height ):
            temp = target_height
            target_height = target_width
            target_width = temp

        image.thumbnail(
            size = (target_width, target_height),
            resample=PIL.Image.Resampling.LANCZOS  # type: ignore
        )
        image = convert_image(image)

        diffusion_result: DiffusionRunner.Result = DiffusionRunner.run_img2img_and_upload(
            self.pipe,
            ctx,
            args_ctx,
            init_image = image,
            generator=self.generator
        )

        reply_message = "\ntime: " + diffusion_result['time_took']
        

        logging.info(f'building reply text')

        def detect_args_and_print(args_name):
            if args_ctx.proc_kwargs is not None and args_name in args_ctx.proc_kwargs:
                return '\n' + f'{args_name}: {args_ctx.proc_kwargs[args_name]}'
            else:
                return ''

        reply_message += detect_args_and_print('num_inference_steps')
        reply_message += detect_args_and_print('guidance_scale')
        reply_message += detect_args_and_print('strength')

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
