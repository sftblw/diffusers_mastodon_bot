import enum
import io
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
import torch
from torch import autocast

from diffusers_mastodon_bot.bot_request_handlers.bot_request_handler import BotRequestHandler
from diffusers_mastodon_bot.bot_request_handlers.bot_request_context import BotRequestContext
from diffusers_mastodon_bot.bot_request_handlers.diffusion_runner import DiffusionRunner
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_message import DiffuseGameMessages, \
    diffusion_game_message_defaults
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_status import DiffuseGameStatus
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_submission import DiffuseGameSubmission
from diffusers_mastodon_bot.bot_request_handlers.proc_args_context import ProcArgsContext
from diffusers_mastodon_bot.community_pipeline.lpw_stable_diffusion import get_weighted_text_embeddings
from diffusers_mastodon_bot.utils import image_grid

import statistics
import math
from threading import Timer


def format_score(score: float):
    return f'{round(score * 100)}%'


class DiffuseGameHandler(BotRequestHandler):
    class RequestType(enum.Enum):
        NewGame = 1
        AnswerSubmission = 2

    def __init__(self,
                 pipe: diffusers.pipelines.StableDiffusionPipeline,
                 tag_name: str = 'diffuse_game',
                 response_duration_sec: float = 60 * 1,
                 score_early_end_condition = 0.85,
                 messages: Optional[DiffuseGameMessages] = None
                 ):

        self.pipe: diffusers.pipelines.StableDiffusionPipeline = pipe
        self.tag_name = tag_name
        self.re_strip_special_token = re.compile('<\|.*?\|>')

        self.response_duration_sec: float = response_duration_sec

        self.score_early_end_condition = score_early_end_condition

        self.messages: DiffuseGameMessages = diffusion_game_message_defaults(messages)

        self.current_game: Optional[DiffuseGameStatus] = None
        self.current_game_timer: Optional[Timer] = None

    def is_eligible_for(self, ctx: BotRequestContext) -> bool:
        # new game
        is_new_game = (
                ctx.contains_tag_name(self.tag_name)
                and ctx.not_from_self()
        )

        if is_new_game:
            ctx.set_payload(type(self), "req_type", DiffuseGameHandler.RequestType.NewGame)
            return True

        # reply to game
        is_reply_to_game = (
            self.current_game is not None
            and ctx.status.in_reply_to_id == self.current_game.status.id
        )

        is_reply_to_game_reply = (
            self.current_game is not None
            and ctx.status.in_reply_to_id in self.current_game.eligible_status_ids_for_reply
        )

        if is_reply_to_game or (is_reply_to_game_reply and ctx.not_from_self()):
            ctx.set_payload(type(self), "req_type", DiffuseGameHandler.RequestType.AnswerSubmission)
            return True

        return False

    def respond_to(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        reply_type: DiffuseGameHandler.RequestType = ctx.get_payload(type(self), 'req_type')  # type: ignore

        if reply_type == DiffuseGameHandler.RequestType.NewGame:
            self.handle_new_game(ctx, args_ctx)
            return True
        elif reply_type == DiffuseGameHandler.RequestType.AnswerSubmission:
            self.handle_answer_submission(ctx, args_ctx)
            return True

        return False

    def close_game(self, any_ctx: BotRequestContext, early_end_status: Optional[Dict[str, Any]] = None):
        logging.info("closing game")

        this_game = self.current_game
        self.current_game = None

        if this_game is None:
            logging.info("this_game is None, early returning.")
            return
        if this_game.status is None:
            logging.info("this_game.status is None, early returning.")
            return

        answer_info = ''
        answer_info += self.messages['question_by'].replace('{account}', '@' + this_game.questioner_acct)
        if this_game.gold_positive_prompt is not None:
            answer_info += '\n\n' + self.messages['gold_positive_prompt'] \
                .replace('{prompt}', this_game.gold_positive_prompt)
        if this_game.gold_negative_prompt is not None:
            answer_info += '\n\n' + self.messages['gold_negative_prompt'] \
                .replace('{prompt}', this_game.gold_negative_prompt)

        if len(this_game.submissions) == 0:
            message = self.messages['game_no_player'] + '\n\n' + answer_info
            any_ctx.reply_to(this_game.status, message[0:480],
                             visibility=any_ctx.bot_ctx.default_visibility, spoiler_text=self.messages['game_no_player_cw'], untag=True)
            return

        scores: List[DiffuseGameSubmission] = \
            sorted(this_game.submissions.values(),
                   key=lambda submission: submission['score'],
                   reverse=True)

        def format_submission(submission: DiffuseGameSubmission):
            return (
                    f'{submission["display_name"][0:20]} '
                    + f'{submission["acct_as_mention"]} '
                    + f'({format_score(submission["score"])})'
            )

        response_body = ''

        if early_end_status is not None:
            response_body += '\n\n' + self.messages['game_early_end']
        else:
            response_body += '\n\n' + self.messages['game_end']

        response_body += '\n\n' + self.messages['game_winner'].replace('{winner}', format_submission(scores[0]))

        if len(scores) >= 1:
            response_body += '\n\n'
            response_body += '\n'.join(
                [f'rank {i + 2}. {format_submission(submission)}' for i, submission in enumerate(scores[1:])]
            )

        response_body = response_body.strip()
        result_status = any_ctx.mastodon.status_post(response_body[0:480], visibility=any_ctx.bot_ctx.default_visibility, in_reply_to_id=this_game.status['id'])

        any_ctx.mastodon.status_post(
            answer_info[0:480],
            in_reply_to_id=result_status['id'],
            visibility=any_ctx.bot_ctx.default_visibility,
            spoiler_text=self.messages['answer_submission_was_by_cw'],
            )

    def handle_new_game(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        if self.current_game is not None:
            ctx.reply_to(ctx.status, self.messages['new_game_already_exists'])
            return True  # it is correctly processed case.

        if ctx.status["visibility"] != "direct":
            ctx.reply_to(ctx.status, self.messages['new_game_should_be_direct'])
            return True  # it is correctly processed case.

        if args_ctx.prompts['positive'] is None or len(args_ctx.prompts['positive']) == 0:
            ctx.reply_to(ctx.status, self.messages['new_game_prompt_is_missing'])
            return True  # it is correctly processed case.

        # start
        in_progress_status = ctx.reply_to(ctx.status, 'processing...', visibility="direct")

        in_progress_public_status = ctx.mastodon.status_post(
            self.messages['new_game_generation_in_progress'],
            visibility=ctx.bot_ctx.default_visibility
        )

        diffusion_result: DiffusionRunner.Result = \
            DiffusionRunner.run_diffusion_and_upload(self.pipe, ctx, args_ctx)

        media_ids = [image_posted['id'] for image_posted in diffusion_result["images_list_posted"]]
        if diffusion_result["has_any_nsfw"] and ctx.bot_ctx.no_image_on_any_nsfw:
            media_ids = None

        current_game_status: Dict[str, Any] = ctx.mastodon.status_post(
            self.messages['new_game_start_announce'],
            media_ids=media_ids,
            visibility=ctx.bot_ctx.default_visibility,
            sensitive=True,
            in_reply_to_id=in_progress_public_status['id']
        )

        def calc_weighted_embeddings(positive: str, negative: Optional[str]):
            text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
                pipe=self.pipe,
                prompt=positive,
                uncond_prompt=negative,
                max_embeddings_multiples=3,
            )
            return text_embeddings, uncond_embeddings

        self.current_game = DiffuseGameStatus(
            status=current_game_status,
            questioner_url=ctx.status['account']['url'],
            questioner_acct=ctx.status['account']['acct'],
            positive_prompt=args_ctx.prompts['positive'],
            negative_prompt=args_ctx.prompts['negative'],
            calc_weighted_embeddings=calc_weighted_embeddings
        )

        self.current_game_timer = Timer(self.response_duration_sec, self.close_game, args=[ctx])
        self.current_game_timer.start()

        ctx.reply_to(in_progress_status, self.messages['new_game_start_success']
                     + '\n\n' + self.current_game.status['url'], spoiler_text='', visibility='direct')

        logging.info(f'sent')

        return True

    def handle_answer_submission(self, ctx: BotRequestContext, args_ctx: ProcArgsContext):
        if self.current_game is None:
            ctx.reply_to(ctx.status, self.messages['answer_submission_game_does_not_exist'])
            return True  #

        submitter_url = ctx.status['account']['url']

        if self.current_game.questioner_url == submitter_url:
            ctx.reply_to(ctx.status, self.messages['answer_submission_is_done_by_questioner'], visibility="direct")
            return True  #

        left_chance: int = self.current_game.left_chance_for(submitter_url)

        if left_chance <= 0:
            ctx.reply_to(ctx.status, self.messages['answer_submission_no_chances_left'])
            return True

        current_submission: DiffuseGameSubmission = self.current_game.set_submission(
            ctx.status,
            args_ctx.prompts['positive'],
            args_ctx.prompts['negative']
        )

        left_chance -= 1

        cur_score = current_submission['score']

        if cur_score < self.score_early_end_condition:
            message = ''

            if left_chance >= 2:
                message = self.messages['answer_submission_left_chance_many']
            elif left_chance == 1:
                message = self.messages['answer_submission_left_chance_last']
            elif left_chance <= 0:
                message = self.messages['answer_submission_left_chance_none']

            message = (
                message
                .replace('{score}', format_score(cur_score))
                .replace('{chance_count}', str(left_chance))
            )

            status = ctx.reply_to(ctx.status, message)

            self.current_game.register_status_as_eligible_for_reply(ctx.status)
            self.current_game.register_status_as_eligible_for_reply(status)

        else:
            ctx.reply_to(ctx.status,
                         self.messages['answer_submission_perfect']
                            .replace('{score}', format_score(cur_score))
                            .replace('{score_early_end_condition}', format_score(self.score_early_end_condition))
            )
            self.current_game_timer.cancel()
            self.close_game(any_ctx=ctx, early_end_status=ctx.status)
