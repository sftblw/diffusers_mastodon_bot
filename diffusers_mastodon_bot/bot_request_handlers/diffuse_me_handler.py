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
from .proc_args_context import ProcArgsContext
from ..utils import image_grid


class DiffuseMeHandler(BotRequestHandler):
    def __init__(self, pipe: diffusers.pipelines.StableDiffusionPipeline, tag_name: str='diffuse_me'):
        self.pipe = pipe
        self.tag_name = tag_name
        self.re_strip_special_token = re.compile('<\|.*?\|>')

    def is_eligible_for(self, ctx: BotRequestContext) -> bool:
        return ctx.contains_tag_name(self.tag_name) and (
                ctx.mentions_bot() or ctx.is_self_response
        )

    def respond_to(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        # start
        in_progress_status = self.reply_in_progress(ctx, args_ctx)

        time_took = 0

        has_any_nsfw = False
        generated_images_raw_pil = []

        with autocast(ctx.bot_ctx.device_name):
            start_time = time.time()
            filename_root = datetime.now().strftime('%Y-%m-%d') + f'_{str(start_time)}'

            left_images_count = args_ctx.target_image_count

            while left_images_count > 0:
                cur_process_count = min(ctx.bot_ctx.max_batch_process, left_images_count)
                logging.info(f"processing {args_ctx.target_image_count - left_images_count + 1} of {args_ctx.target_image_count}, "
                             + f"by {cur_process_count}")

                pipe_results = self.pipe(
                    [args_ctx.prompts['positive']] * cur_process_count,
                    negative_prompt=([args_ctx.prompts['negative_with_default']] * cur_process_count
                                     if args_ctx.prompts['negative_with_default'] is not None
                                     else None),
                    **args_ctx.proc_kwargs
                )

                generated_images_raw_pil.extend(pipe_results.images)
                if pipe_results.nsfw_content_detected:
                    has_any_nsfw = True

                left_images_count -= ctx.bot_ctx.max_batch_process

            end_time = time.time()

            time_took = end_time - start_time
            time_took = int(time_took * 1000) / 1000

            reply_message = f'took: {time_took}s'

            # save anyway
            for idx in range(args_ctx.target_image_count):
                image_filename = str(Path(ctx.bot_ctx.output_save_path, filename_root + f'_{idx}' + '.png').resolve())
                text_filename = str(Path(ctx.bot_ctx.output_save_path, filename_root + f'_{idx}' + '.txt').resolve())

                time_took += end_time - start_time

                image: Image = generated_images_raw_pil[idx]

                image.save(image_filename, "PNG")
                Path(text_filename).write_text(json.dumps(args_ctx.prompts))

        if ctx.bot_ctx.delete_processing_message:
            ctx.mastodon.status_delete(in_progress_status['id'])

        logging.info(f'preparing images to upload')

        image_grid_unit = int(ctx.bot_ctx.image_tile_xy[0] * ctx.bot_ctx.image_tile_xy[1])
        pil_image_grids = []
        if args_ctx.target_image_count == 1:
            pil_image_grids = generated_images_raw_pil
        else:
            for i in range(0, len(generated_images_raw_pil), image_grid_unit):
                cur_pil_image_slice = generated_images_raw_pil[i: i + image_grid_unit]

                image_slice_len = len(cur_pil_image_slice)

                max_x = ctx.bot_ctx.image_tile_xy[0] \
                    if image_slice_len >= ctx.bot_ctx.image_tile_xy[0] \
                    else ctx.bot_ctx.image_tile_xy[0] % len(cur_pil_image_slice)
                max_y = math.ceil(image_slice_len / ctx.bot_ctx.image_tile_xy[0])

                fitting_square = math.pow(math.floor(math.sqrt(image_grid_unit)), 2)

                if fitting_square > max_x * max_y:
                    max_x = fitting_square
                    max_y = max_y

                grid_image = image_grid(cur_pil_image_slice, max_x, max_y)
                pil_image_grids.append(grid_image)

        logging.info(f'uploading {len(generated_images_raw_pil)} images')

        images_list_posted: List[Dict[str, any]] = []

        for pil_image in pil_image_grids:
            try:
                # pil to png
                # https://stackoverflow.com/a/33117447/4394750
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                png_bytes = img_byte_arr.getvalue()

                upload_result = ctx.mastodon.media_post(png_bytes, 'image/png')
                images_list_posted.append(upload_result)
            except Exception as ex:
                logging.error(f'error on image upload: {ex}')
                pass

        logging.info(f'building reply text')

        def detect_args_and_print(args_name):
            if args_ctx.proc_kwargs is not None and args_name in args_ctx.proc_kwargs:
                return '\n' + f'{args_name}: {args_ctx.proc_kwargs[args_name]}'
            else:
                return ''

        reply_message += detect_args_and_print('num_inference_steps')
        reply_message += detect_args_and_print('guidance_scale')

        if has_any_nsfw:
            reply_message += '\n\n' + 'nsfw content detected, some of result will be a empty image'

        reply_message += '\n\n' + f'prompt: \n{args_ctx.prompts["positive"]}'

        if args_ctx.prompts['negative'] is not None:
            reply_message += '\n\n' + f'negative prompt (without default): \n{args_ctx.prompts["negative"]}'

        if len(reply_message) >= 450:
            reply_message = reply_message[0:400] + '...'

        media_ids = [image_posted['id'] for image_posted in images_list_posted]
        if has_any_nsfw and ctx.bot_ctx.no_image_on_any_nsfw:
            media_ids = None

        reply_target_status = ctx.status if ctx.bot_ctx.delete_processing_message else in_progress_status
        ctx.mastodon.status_reply(reply_target_status, reply_message,
                                   media_ids=media_ids,
                                   visibility=ctx.reply_visibility,
                                   spoiler_text='[done] ' + args_ctx.prompts['positive'][0:20] + '...',
                                   sensitive=True
                                   )

        logging.info(f'sent')

        return True

    def reply_in_progress(self, ctx: BotRequestContext, args_ctx: ProcArgsContext):
        # start message
        processing_body = ''

        # noinspection PyUnresolvedReferences
        tokenizer: transformers.CLIPTokenizer = self.pipe.tokenizer

        decoded_ids = tokenizer(args_ctx.prompts["positive"])['input_ids'][0:76]
        decoded_text = tokenizer.decode(decoded_ids)
        decoded_text = self.re_strip_special_token.sub('', decoded_text).strip()

        if decoded_text != args_ctx.prompts["positive"]:
            processing_body += f'\n\nencoded prompt is: {decoded_text[0:400]}'

        in_progress_status = ctx.reply_to(status=ctx.status, body=processing_body, spoiler_text='processing...')
        return in_progress_status
