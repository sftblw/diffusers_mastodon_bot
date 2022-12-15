import logging
import logging
import re
from typing import *

from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion \
    import StableDiffusionLongPromptWeightingPipeline as StableDiffusionLpw
from .bot_request_context import BotRequestContext
from .bot_request_handler import BotRequestHandler
from .diffusion_runner import DiffusionRunner
from .proc_args_context import ProcArgsContext

logger = logging.getLogger(__name__)


class DiffuseMeHandler(BotRequestHandler):
    def __init__(self,
                 pipe: StableDiffusionLpw,
                 tag_name: str = 'diffuse_me',
                 allow_self_request_only: bool = False
                 ):
        self.pipe = pipe
        self.tag_name = tag_name
        self.allow_self_request_only = allow_self_request_only
        self.re_strip_special_token = re.compile(r'<\|.*?\|>')

    def is_eligible_for(self, ctx: BotRequestContext) -> bool:
        contains_hash = ctx.contains_tag_name(self.tag_name)
        if not contains_hash:
            return False

        return (
                (ctx.mentions_bot() and ctx.not_from_self() and not self.allow_self_request_only)
                or
                not ctx.not_from_self()
        )

    def respond_to(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        # start
        positive_input_form = args_ctx.prompts['positive']
        negative_input_form = args_ctx.prompts['negative']

        in_progress_status = ctx.reply_to(ctx.status, 'processing...', keep_context=False)

        if 'num_inference_steps' in args_ctx.proc_kwargs \
                and args_ctx.proc_kwargs['num_inference_steps'] is not None:
            args_ctx.proc_kwargs['num_inference_steps'] = int(args_ctx.proc_kwargs['num_inference_steps'])

        diffusion_result: DiffusionRunner.Result = DiffusionRunner.run_diffusion_and_upload(self.pipe, ctx, args_ctx)

        logger.info(f'building reply text')

        reply_message, spoiler_text, media_ids = DiffusionRunner.make_reply_message_contents(
            ctx,
            args_ctx,
            diffusion_result,
            detecting_args=['num_inference_steps', 'guidance_scale'],
            positive_input_form=positive_input_form,
            negative_input_form=negative_input_form
        )

        behavior_conf = ctx.bot_ctx.behavior_conf

        reply_target_status = ctx.status if behavior_conf.delete_processing_message else in_progress_status

        replied_status = ctx.reply_to(
            reply_target_status,
            reply_message,
            media_ids=media_ids,
            visibility=ctx.reply_visibility,
            spoiler_text=spoiler_text,
            sensitive=True,
            tag_behind=behavior_conf.tag_behind_on_image_post
        )

        if behavior_conf.tag_behind_on_image_post:
            ctx.mastodon.status_reblog(replied_status['id'])

        if behavior_conf.delete_processing_message:
            ctx.mastodon.status_delete(in_progress_status)

        logger.info(f'sent')

        return True

    def reply_in_progress(self, ctx: BotRequestContext, args_ctx: ProcArgsContext, positive_input_form: str,
                          negative_input_form: Optional[str]):
        processing_body = DiffusionRunner.make_processing_body(args_ctx, positive_input_form, negative_input_form)
        in_progress_status = ctx.reply_to(status=ctx.status,
                                          body=processing_body if len(processing_body) > 0 else 'processing...',
                                          spoiler_text='processing...' if len(processing_body) > 0 else None,
                                          keep_context=True if len(processing_body) > 0 else False
                                          )
        return in_progress_status
