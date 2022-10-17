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
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_message import DiffuseGameMessages
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_status import DiffuseGameStatus
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_submission import DiffuseGameSubmission
from diffusers_mastodon_bot.bot_request_handlers.proc_args_context import ProcArgsContext
from diffusers_mastodon_bot.utils import image_grid

import statistics
import math
from threading import Timer


class DiffuseGameHandler(BotRequestHandler):
    class RequestType(enum.Enum):
        NewGame = 1
        AnswerSubmission = 2

    def __init__(self,
                 pipe: diffusers.pipelines.StableDiffusionPipeline,
                 tag_name: str = 'diffuse_game',
                 response_duration_sec: float = 60 * 1,
                 messages: Optional[DiffuseGameMessages] = None
                 ):

        self.pipe: diffusers.pipelines.StableDiffusionPipeline = pipe
        self.tag_name = tag_name
        self.re_strip_special_token = re.compile('<\|.*?\|>')

        self.response_duration_sec: float = response_duration_sec

        if messages is None:
            messages = {
                "new_game_already_exists":
                    "A diffusion game is already in progress.",
                "new_game_should_be_direct":
                    "To create new game, send me # diffuse_game with \"direct\" message (aka private).",
                "new_game_prompt_is_missing":
                    "Your prompt is missing! Please provide me # diffuse_me with prompt to generate an image, with DM.",
                "new_game_start_announce":
                    "Diffusion guessing game is started! guess the prompt and reply to this message! \n\n#bot_message",
                "new_game_start_success":
                    "ok! new game is generated! (You are not allowed to submit guesses, though!)",
                "answer_submission_game_does_not_exist":
                    "This game does not exist anymore!",
                "answer_submission_is_done_by_submitter":
                    "You are not allowed to answer to your question. Anyway You know it, isn't it?",
                "answer_submission_was_by_cw":
                    "Answer was...",
                "game_no_player":
                    "no players joined guessing game this time...",
                "game_no_player_cw":
                    "no player joined...",
                "game_end":
                    "Game end!",
                "game_winner":
                    "{winner} is the final winner!",

                "question_by": "This question was prompted by {account}!",
                "gold_positive_prompt": "prompt was:\n{prompt}",
                "gold_negative_prompt": "negative prompt was:\n{prompt}",
            }
        self.messages: DiffuseGameMessages = messages

        self.current_game: Optional[DiffuseGameStatus] = None
        self.current_game_timer: Optional[Timer] = None

    def is_eligible_for(self, ctx: BotRequestContext) -> bool:
        # new game
        is_new_game = (
                ctx.contains_tag_name(self.tag_name)
                and ctx.mentions_bot()
        )

        if is_new_game:
            ctx.set_payload(type(self), "req_type", DiffuseGameHandler.RequestType.NewGame)
            return True

        # reply to game
        is_reply_to_game = (
                self.current_game is not None
                and ctx.status.in_reply_to_id == self.current_game.status.id
        )

        if is_reply_to_game:
            ctx.set_payload(type(self), "req_type", DiffuseGameHandler.RequestType.AnswerSubmission)
            return True

        return False

    def close_game(self, new_game_ctx: BotRequestContext):
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
        answer_info += self.messages['question_by'].replace('{account}', '@' + this_game.submitter_acct)
        if this_game.gold_positive_prompt is not None:
            answer_info += '\n\n' + self.messages['gold_positive_prompt'] \
                .replace('{prompt}', this_game.gold_positive_prompt)
        if this_game.gold_negative_prompt is not None:
            answer_info += '\n\n' + self.messages['gold_negative_prompt'] \
                .replace('{prompt}', this_game.gold_negative_prompt)

        if len(this_game.submissions) == 0:
            message = self.messages['game_no_player'] + '\n\n' + answer_info
            new_game_ctx.reply_to(this_game.status, message[0:480],
                                  visibility="unlisted", spoiler_text=self.messages['game_no_player_cw'], untag=True)
            return

        scores: List[DiffuseGameSubmission] = \
            sorted(this_game.submissions.values(),
                   key=lambda submission: submission['score'],
                   reverse=True)

        def format_submission(submission: DiffuseGameSubmission):
            return (
                    f'{submission["display_name"][0:20]} '
                    + f'{submission["acct_as_mention"]} '
                    + f'({math.floor(submission["score"] * 100)}%)'
            )

        response_body = self.messages['game_end']
        response_body += '\n\n' + self.messages['game_winner'].replace('{winner}', format_submission(scores[0]))

        if len(scores) >= 1:
            response_body += '\n\n'
            response_body += '\n'.join(
                [f'rank {i + 2}. {format_submission(submission)}' for i, submission in enumerate(scores[1:])]
            )

        result_status = new_game_ctx.reply_to(this_game.status, response_body[0:480], visibility="unlisted", untag=True)

        new_game_ctx.reply_to(result_status,
                              answer_info[0:480],
                              visibility="unlisted",
                              spoiler_text=self.messages['answer_submission_was_by_cw'],
                              untag=True)

    def respond_to(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        reply_type: DiffuseGameHandler.RequestType = ctx.get_payload(type(self), 'req_type')  # type: ignore

        if reply_type == DiffuseGameHandler.RequestType.NewGame:
            self.handle_new_game(ctx, args_ctx)
            return True
        elif reply_type == DiffuseGameHandler.RequestType.AnswerSubmission:
            self.handle_answer_submission(ctx, args_ctx)
            return True

        return False

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
        in_progress_status = self.reply_in_progress(ctx, args_ctx)

        diffusion_result: DiffusionRunner.Result = \
            DiffusionRunner.run_diffusion_and_upload(self.pipe, ctx, args_ctx)

        media_ids = [image_posted['id'] for image_posted in diffusion_result["images_list_posted"]]
        if diffusion_result["has_any_nsfw"] and ctx.bot_ctx.no_image_on_any_nsfw:
            media_ids = None

        current_game_status: Dict[str, Any] = ctx.mastodon.status_post(
            self.messages['new_game_start_announce'],
            media_ids=media_ids,
            visibility='unlisted',
            sensitive=True
        )

        self.current_game = DiffuseGameStatus(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            status=current_game_status,
            submitter_url=ctx.status['account']['url'],
            submitter_acct=ctx.status['account']['acct'],
            positive_prompt=args_ctx.prompts['positive'],
            negative_prompt=args_ctx.prompts['negative']
        )

        self.current_game_timer = Timer(self.response_duration_sec, self.close_game, args=[ctx])
        self.current_game_timer.start()

        ctx.reply_to(in_progress_status, self.messages['new_game_start_success']
                     + '\n\n' + self.current_game.status['url'], spoiler_text='')

        logging.info(f'sent')

        return True

    def handle_answer_submission(self, ctx: BotRequestContext, args_ctx: ProcArgsContext):
        if self.current_game is None:
            ctx.reply_to(ctx.status, self.messages['answer_submission_game_does_not_exist'])
            return True  #

        if self.current_game.submitter_url == ctx.status['account']['url']:
            ctx.reply_to(ctx.status, self.messages['answer_submission_is_done_by_submitter'], visibility="direct")
            return True  #

        self.current_game.set_submission(ctx.status, args_ctx.prompts['positive'], args_ctx.prompts['negative'])
        ctx.mastodon.status_favourite(ctx.status)

    def reply_in_progress(self, ctx: BotRequestContext, args_ctx: ProcArgsContext):
        processing_body = DiffusionRunner.make_processing_body(self.pipe, args_ctx=args_ctx)
        in_progress_status = ctx.reply_to(status=ctx.status,
                                          body=processing_body if len(processing_body) > 0 else 'processing...',
                                          spoiler_text='processing...' if len(processing_body) > 0 else None,
                                          visibility="direct")
        return in_progress_status
