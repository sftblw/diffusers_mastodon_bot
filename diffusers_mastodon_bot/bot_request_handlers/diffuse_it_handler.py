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
        positive_input_form, negative_input_form = DiffusionRunner.args_prompts_as_input_text(self.pipe, args_ctx)
        
        in_progress_status = self.reply_in_progress(ctx, args_ctx, positive_input_form, negative_input_form)

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
            # make it as square
            target_width = min(target_width, target_height)
            target_height = min(target_width, target_height)
        elif ( image.width > image.height and target_width < target_height ) \
            or ( image.width < image.height and target_width > target_height ):
            # fit orientation
            temp = target_height
            target_height = target_width
            target_width = temp

        # upscale too small image, to fit in
        if image.width < target_width and image.height < target_height:
            width_ratio = target_width / image.width
            height_ratio = target_height / image.height

            if width_ratio > height_ratio:
                # based on height
                image = image.resize(
                    (int(target_height * (image.width / image.height)), int(target_height)),
                    resample=PIL.Image.Resampling.LANCZOS
                )
            elif width_ratio <= height_ratio:
                # based on width:
                image = image.resize(
                    (int(target_width), int(target_width * (image.height / image.width))),
                    resample=PIL.Image.Resampling.LANCZOS
                )

        # fit in the image
        image.thumbnail(
            size = (target_width, target_height),
            resample=PIL.Image.Resampling.LANCZOS  # type: ignore
        )
        image = convert_image(image)

        # increase steps by strength, to run like default
        num_inference_steps_original = None
        if 'num_inference_steps' in args_ctx.proc_kwargs \
            and args_ctx.proc_kwargs['num_inference_steps'] is not None:
            strength = args_ctx.proc_kwargs['strength'] if 'strength' in args_ctx.proc_kwargs else None
            if strength is None:
                strength = 0.8  # pipeline default
            if strength > 0:
                num_inference_steps_original = int(args_ctx.proc_kwargs['num_inference_steps'])
                args_ctx.proc_kwargs['num_inference_steps'] = int(num_inference_steps_original / strength)

        diffusion_result: DiffusionRunner.Result = DiffusionRunner.run_img2img_and_upload(
            self.pipe,
            ctx,
            args_ctx,
            init_image = image,
            generator=self.generator
        )

        logging.info(f'building reply text')

        reply_message, spoiler_text, media_ids = DiffusionRunner.make_reply_message_contents(
            ctx,
            args_ctx,
            diffusion_result,
            detecting_args=['guidance_scale', 'strength'],
            args_custom_text=f'args.num_inference_steps (actual): {args_ctx.proc_kwargs["num_inference_steps"]} (input: {num_inference_steps_original})' \
                if num_inference_steps_original is not None \
                else None,
            positive_input_form=positive_input_form,
            negative_input_form=negative_input_form
        )

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

    def reply_in_progress(self, ctx: BotRequestContext, args_ctx: ProcArgsContext, positive_input_form: str, negative_input_form: Optional[str]):
        processing_body = DiffusionRunner.make_processing_body(args_ctx, positive_input_form, negative_input_form)
        in_progress_status = ctx.reply_to(status=ctx.status,
                                          body=processing_body if len(processing_body) > 0 else 'processing...',
                                          spoiler_text='processing...' if len(processing_body) > 0 else None,
                                          )
        return in_progress_status
