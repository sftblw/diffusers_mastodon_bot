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
import traceback

import diffusers.pipelines
import torch
import transformers
import PIL
import PIL.PngImagePlugin
from torch import autocast
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .bot_request_handler import BotRequestHandler
from .bot_request_context import BotRequestContext
from .proc_args_context import ProcArgsContext
from ..utils import image_grid

from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion \
    import StableDiffusionLongPromptWeightingPipeline as StableDiffusionLpw


logger = logging.getLogger(__name__)


class DiffusionRunner:
    class Result(TypedDict):
        image_filenames: List[str]
        images_list_posted: List[Any]
        has_any_nsfw: bool
        time_took: str

    re_strip_special_token = re.compile('<\|.*?\|>')

    @staticmethod
    def tokenize_prompt(
            prompt: str,
            tokenizer: CLIPTokenizer,
            ) -> torch.Tensor:  # torch.Size([1, 77])

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        # torch.Size([1, 77])
        return text_inputs.input_ids.squeeze(0)[0:77].unsqueeze(0)

    @staticmethod
    def prompt_as_input_text(prompt: str, tokenizer: CLIPTokenizer) -> str:
        text_input_ids = DiffusionRunner.tokenize_prompt(prompt, tokenizer)
        # torch.Size([1, 77])
        text_input_ids = text_input_ids[0]
        # torch.Size([77])
        decoded_text = tokenizer.decode(text_input_ids)
        decoded_text: str = DiffusionRunner.re_strip_special_token.sub('', decoded_text).strip()
        return decoded_text

    @staticmethod
    def embed_tokens(tokens: torch.Tensor, text_encoder: CLIPTextModel) -> torch.Tensor:
        # https://github.com/huggingface/diffusers/blob/v0.4.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
        # torch.Size([1, 77])
        embedding: BaseModelOutputWithPooling = text_encoder(tokens.to(text_encoder.device))
        # torch.Size([1, 77, 768])
        return embedding.last_hidden_state

    @staticmethod
    def embed_prompt(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel):
        tokenized = DiffusionRunner.tokenize_prompt(prompt, tokenizer)
        # torch.Size([1, 77])
        embed = DiffusionRunner.embed_tokens(tokenized, text_encoder)
        # torch.Size([1, 77, 768])

        return embed

    @staticmethod
    def make_processing_body(
        args_ctx: ProcArgsContext,
        positive_input_form: str,
        negative_input_form: Optional[str]
    ) -> str:
        # start message
        processing_body = ''

        if positive_input_form != args_ctx.prompts["positive"]:
            processing_body += f'\n\npositive prompt:\n{positive_input_form}'

        if args_ctx.prompts['negative'] is not None and len(args_ctx.prompts['negative']) > 0:
            if negative_input_form != args_ctx.prompts["negative"]:
                processing_body += f'\n\nnegative prompt:\n{negative_input_form}'

        return processing_body[0:400]

    @staticmethod
    def run_sth_and_upload(
        ctx: BotRequestContext,
        args_ctx: ProcArgsContext,
        pipe: Any,
        filename_root: str,
        run_diffusion_fn: Callable,
        run_diffusion_fn_kwargs: Dict['str', Any] = {},
    ) -> Result:

        result: DiffusionRunner.Result = {
            "image_filenames": [],
            "images_list_posted": [],
            "has_any_nsfw": False,
            "time_took": ''
        }

        with autocast(ctx.bot_ctx.device_name):
            start_time = time.time()
            
            generated_images_raw_pil, has_any_nsfw = run_diffusion_fn(ctx, args_ctx, pipe, **run_diffusion_fn_kwargs)
            result["has_any_nsfw"] = has_any_nsfw
            
            end_time = time.time()

            time_took = end_time - start_time
            time_took = int(time_took * 1000) / 1000

            result["time_took"] = f'{time_took}s'

        if ctx.bot_ctx.save_image:
            save_result = DiffusionRunner.save_images(
                ctx, args_ctx, filename_root, generated_images_raw_pil,
                save_args=ctx.bot_ctx.save_args, save_args_text=ctx.bot_ctx.save_args_text
            )
            result["image_filenames"] = save_result

        uploaded_images = DiffusionRunner.upload_images(ctx, generated_images_raw_pil)
        result["images_list_posted"] = uploaded_images

        return result

    @staticmethod
    def run_diffusion_and_upload(pipe: diffusers.pipelines.StableDiffusionPipeline,
                                 ctx: BotRequestContext,
                                 args_ctx: ProcArgsContext) -> Result:
        return DiffusionRunner.run_sth_and_upload(
            ctx,
            args_ctx,
            pipe,
            filename_root=datetime.now().strftime('%Y-%m-%d_%H-%M-%S_sd'),
            run_diffusion_fn=DiffusionRunner.run_diffusion,
        )

    @staticmethod
    def run_diffusion(ctx, args_ctx, pipe: StableDiffusionLpw) -> Tuple[List[PIL.Image.Image], bool]:
        left_images_count = args_ctx.target_image_count
        generated_images_raw_pil = []
        has_any_nsfw = False

        def key_or_none(key):
            return args_ctx.proc_kwargs[key] if key in args_ctx.proc_kwargs else None

        manual_proc_kwargs = {
            "width": key_or_none('width'),
            "height": key_or_none('height'),
            "num_inference_steps": key_or_none('num_inference_steps'),
            "guidance_scale": key_or_none('guidance_scale')
        }

        while left_images_count > 0:
            cur_process_count = min(ctx.bot_ctx.max_batch_process, left_images_count)
            logger.info(
                f"processing {args_ctx.target_image_count - left_images_count + 1} of {args_ctx.target_image_count}, "
                + f"by {cur_process_count}")

            pipe_results = pipe.text2img(
                [args_ctx.prompts['positive']] * cur_process_count,
                negative_prompt=([args_ctx.prompts['negative_with_default']] * cur_process_count
                                 if args_ctx.prompts['negative_with_default'] is not None
                                 else None),
                **manual_proc_kwargs
            )

            generated_images_raw_pil.extend(pipe_results.images)
            if pipe_results.nsfw_content_detected:
                has_any_nsfw = True

            left_images_count -= ctx.bot_ctx.max_batch_process

        return generated_images_raw_pil, has_any_nsfw

    @staticmethod
    def run_img2img_and_upload(pipe: diffusers.pipelines.StableDiffusionImg2ImgPipeline,
                                ctx: BotRequestContext,
                                args_ctx: ProcArgsContext,
                                init_image: PIL.Image.Image,
                                generator: Optional[torch.Generator] = None
                                ) -> Result:
        filename_root = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_im2im')

        result = DiffusionRunner.run_sth_and_upload(
            ctx,
            args_ctx,
            pipe,
            filename_root=filename_root,
            run_diffusion_fn=DiffusionRunner.run_img2img,
            run_diffusion_fn_kwargs={
                "init_image": init_image,
                "generator": generator
            }
        )

        # save init image too
        if ctx.bot_ctx.save_image:
            DiffusionRunner.save_images(
                ctx,
                args_ctx,
                filename_root=filename_root + '_src',
                generated_images_raw_pil=[init_image],
                save_args=False
            )

        return result

    @staticmethod
    def run_img2img(ctx, args_ctx, pipe: StableDiffusionLpw, init_image: PIL.Image.Image, generator: Optional[torch.Generator] = None) -> Tuple[List[PIL.Image.Image], bool]:
        left_images_count = args_ctx.target_image_count
        generated_images_raw_pil = []
        has_any_nsfw = False

        def key_or_none(key):
            return args_ctx.proc_kwargs[key] if key in args_ctx.proc_kwargs else None

        manual_proc_kwargs = {
            "width": key_or_none('width'),
            "height": key_or_none('height'),
            "num_inference_steps": key_or_none('num_inference_steps'),
            "guidance_scale": key_or_none('guidance_scale'),
            "strength": key_or_none('strength'),
        }

        if manual_proc_kwargs['strength'] is None:
            del manual_proc_kwargs['strength']

        while left_images_count > 0:
            cur_process_count = min(ctx.bot_ctx.max_batch_process, left_images_count)
            logger.info(
                f"processing {args_ctx.target_image_count - left_images_count + 1} of {args_ctx.target_image_count}, "
                + f"by {cur_process_count}")

            pipe_results = pipe.img2img(
                image=init_image,
                prompt=[args_ctx.prompts['positive']] * cur_process_count,
                negative_prompt=([args_ctx.prompts['negative_with_default']] * cur_process_count
                                    if args_ctx.prompts['negative_with_default'] is not None
                                    else None),
                generator=generator,
                **manual_proc_kwargs
            )

            generated_images_raw_pil.extend(pipe_results.images)
            if pipe_results.nsfw_content_detected:
                has_any_nsfw = True

            left_images_count -= ctx.bot_ctx.max_batch_process

        return generated_images_raw_pil, has_any_nsfw

    @staticmethod
    def save_images(
            ctx: BotRequestContext,
            args_ctx: ProcArgsContext,
            filename_root: str,
            generated_images_raw_pil: List[PIL.Image.Image],
            save_args: bool = True,
            save_args_text: bool = False
    ) -> List[str]:

        image_filenames = []

        info_data: Optional[str] = None
        if save_args:
            info_data_obj = {
                "args_ctx": args_ctx,
                "args_version": "0.1.0"
            }
            info_data = json.dumps(info_data_obj, default=vars)

        # save anyway
        for idx in range(len(generated_images_raw_pil)):
            image_filename = str(Path(ctx.bot_ctx.output_save_path, filename_root + f'_{idx}' + '.png').resolve())

            image: PIL.Image.Image = generated_images_raw_pil[idx]

            png_metadata = None
            if save_args:
                png_metadata = PIL.PngImagePlugin.PngInfo()
                png_metadata.add_text("diffusers_mastodon_bot_args", info_data, zip=True)  # zText

            image.save(image_filename, "PNG", pnginfo=png_metadata)

            image_filenames.append(image_filename)

        if save_args and save_args_text:
            text_filename = str(Path(ctx.bot_ctx.output_save_path, filename_root + '.txt').resolve())
            Path(text_filename).write_text(info_data)

        return image_filenames

    @staticmethod
    def upload_images(ctx: BotRequestContext, generated_images_raw_pil: List[PIL.Image.Image]) -> List[PIL.Image.Image]:

        logger.info(f'preparing images to upload')

        pil_image_grids = DiffusionRunner.image_grid_by_cfg(
            pil_images=generated_images_raw_pil,
            image_tile_x = ctx.bot_ctx.image_tile_xy[0],
            image_tile_y = ctx.bot_ctx.image_tile_xy[1],
            image_tile_auto_expand = ctx.bot_ctx.image_tile_auto_expand,
            max_attachment_count = ctx.bot_ctx.image_max_attachment_count
        )

        logger.info(f'uploading {len(generated_images_raw_pil)} images')

        posted_images = []

        for pil_image in pil_image_grids:
            try:
                # pil to png
                # https://stackoverflow.com/a/33117447/4394750
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                png_bytes = img_byte_arr.getvalue()

                upload_result = ctx.mastodon.media_post(png_bytes, 'image/png')
                posted_images.append(upload_result)
            except Exception as ex:
                logger.error(f'error on image upload:\n' + "\n  ".join(traceback.format_exception(ex)))
                pass

        return posted_images

    @staticmethod
    def image_grid_by_cfg(
            pil_images: List[PIL.Image.Image],
            image_tile_x: int,
            image_tile_y: int,
            image_tile_auto_expand: bool,
            max_attachment_count: int
    ):
        """
        :param pil_images: raw PIL images
        :param image_tile_x: columns to tile
        :param image_tile_y: rows to tile
        :param image_tile_auto_expand: changes tile x and y dynamically to spread images across attachment
        :param max_attachment_count: 4 (mastodon)
        :return:
        """
        images_count = len(pil_images)
        if images_count == 1:
            return pil_images

        image_grid_unit = int(image_tile_x * image_tile_y)

        images_grouped: List[List[PIL.Image.Image]] = []

        # maximum square size, smaller than image_tile_x & image_tile_y
        fitting_square = math.pow(math.floor(math.sqrt(image_grid_unit)), 2)

        # spread
        if image_tile_auto_expand and len(pil_images) < max_attachment_count * image_grid_unit:
            for start_i in range(0, max_attachment_count):
                # [0::10] https://stackoverflow.com/a/1403693/4394750
                cur_images_group = pil_images[start_i::max_attachment_count]
                images_grouped.append(cur_images_group)

            first_group_len = len(images_grouped[0])

            fitting_square = math.pow(math.floor(math.sqrt(first_group_len)), 2)

            # x * y = first_group_len
            # => x * x * (ratio=y/x) = first_group_len
            # => x * x = first_group_len * (reverse_ratio=x/y)
            # => x = sqrt ( ^ ), y = x / (x/y)
            x_of_y = image_tile_x / image_tile_y
            image_tile_x = math.sqrt(first_group_len * x_of_y)
            image_tile_y = math.ceil(image_tile_x / x_of_y)
            image_tile_x = math.ceil(image_tile_x)

        # group by predefined grid size
        else:
            for i in range(0, len(pil_images), image_grid_unit):
                cur_images_group = pil_images[i: i + image_grid_unit]
                images_grouped.append(cur_images_group)

        pil_image_grids = []

        for image_group in images_grouped:
            image_slice_len = len(image_group)

            # calculate
            max_y = image_tile_y \
                if image_slice_len >= image_tile_x \
                else image_tile_y % len(image_group)
            max_x = math.ceil(image_slice_len / image_tile_y)

            if fitting_square > max_x * max_y:
                max_x = fitting_square
                max_y = max_y

            grid_image = image_grid(image_group, max_x, max_y)
            pil_image_grids.append(grid_image)
        return pil_image_grids

    @staticmethod
    def make_reply_message_contents(
        ctx: BotRequestContext,
        args_ctx: ProcArgsContext,
        diffusion_result: Any,  # DiffusionRunner.Result
        detecting_args: List[str],
        args_custom_text: Optional[str] = None,
        positive_input_form: str = '',
        negative_input_form: Optional[str] = None
    ):  
        reply_message = "\ntime: " + diffusion_result['time_took']
            
        def detect_args_and_print(args_name):
            if args_ctx.proc_kwargs is not None and args_name in args_ctx.proc_kwargs:
                return '\n' + f'{args_name}: {args_ctx.proc_kwargs[args_name]}'
            else:
                return ''

        for key in detecting_args:
            reply_message += detect_args_and_print(key)

        if args_custom_text is not None:
            reply_message += '\n' + args_custom_text

        if diffusion_result["has_any_nsfw"]:
            reply_message += '\n\n' + 'nsfw content detected, some of result will be a empty image'

        reply_message += '\n\n' + f'prompt: \n{positive_input_form}'

        if negative_input_form is not None:
            reply_message += '\n\n' + f'negative prompt (without default): \n{negative_input_form}'

        if len(reply_message) >= 450:
            reply_message = reply_message[0:400] + '...'

        media_ids = [image_posted['id'] for image_posted in diffusion_result["images_list_posted"]]
        if diffusion_result["has_any_nsfw"] and ctx.bot_ctx.no_image_on_any_nsfw:
            media_ids = None

        spoiler_text = '[done] ' + args_ctx.prompts['positive'][0:20] + '...'

        return reply_message, spoiler_text, media_ids