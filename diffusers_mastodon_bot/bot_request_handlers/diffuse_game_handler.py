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

from .bot_request_handler import BotRequestHandler
from .bot_request_context import BotRequestContext
from .diffusion_runner import DiffusionRunner
from .proc_args_context import ProcArgsContext
from ..utils import image_grid

import statistics
import math
from threading import Timer


class DiffuseGameMessages(TypedDict):
    new_game_already_exists: str
    new_game_should_be_direct: str
    new_game_prompt_is_missing: str
    new_game_start_announce: str
    new_game_start_success: str

    answer_submission_game_does_not_exist: str
    answer_submission_is_done_by_submitter: str

    answer_submission_was_by_cw: str

    game_no_player: str
    game_no_player_cw: str
    game_end: str
    game_winner: str

    question_by: str
    gold_positive_prompt: str
    gold_negative_prompt: str


class DiffuseGameSubmission(TypedDict):
    status: Dict[str, any]
    account_url: str
    acct_as_mention: str
    display_name: str
    positive: Optional[str]
    negative: Optional[str]
    score: float


class DiffusionGameStatus:
    def __init__(self,
                 tokenizer: transformers.CLIPTokenizer,
                 text_encoder: transformers.CLIPTextModel,
                 status: Dict[str, any],
                 submitter_url: str,
                 submitter_acct: str,
                 positive_prompt: str,
                 negative_prompt: Optional[str]
                 ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # status or None
        self.status: Optional[Dict[str, any]] = status
        self.submitter_url = submitter_url
        self.submitter_acct = submitter_acct

        self.gold_positive_prompt: Optional[str] = positive_prompt
        self.gold_negative_prompt: Optional[str] = negative_prompt

        # tokenizer -> embedding -> last layer
        # torch.Size([1, 77, 768])
        self.gold_positive_embedding: Optional[torch.Tensor] = \
            self.prompt_as_embedding(self.gold_positive_prompt) if self.gold_positive_prompt is not None else None

        # torch.Size([768])
        self.gold_positive_embedding_mean = \
            self.gold_positive_embedding[0].mean(dim=0) if self.gold_positive_embedding is not None else None

        self.gold_negative_embedding: Optional[torch.Tensor] = \
            self.prompt_as_embedding(self.gold_negative_prompt) if self.gold_negative_prompt is not None else None

        # torch.Size([768])
        self.gold_negative_embedding_mean = \
            self.gold_negative_embedding[0].mean(dim=0) if self.gold_negative_embedding is not None else None

        # for multiple submission, dictionary with acct is used.
        self.submissions: Dict[str, DiffuseGameSubmission] = {}

    def prompt_as_embedding(self, prompt: str) -> torch.Tensor:
        prompt = DiffusionRunner.embed_prompt(
            prompt=prompt, tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        prompt = prompt.to('cpu')
        return prompt

    def set_submission(self, status: Dict[str, any], positive_prompt: Optional[str], negative_prompt: Optional[str]):
        def get_similarity_score(prompt, gold_prompt, gold_mean):
            if prompt is not None and gold_prompt is not None:
                # torch.Size([1, 77, 768]) -> (indexing 0) -> (77, 768)
                prompt_embedding: torch.Tensor = self.prompt_as_embedding(prompt)[0]
                prompt_mean = prompt_embedding.mean(dim=0)
                # [768]
                similarity: torch.Tensor = \
                    torch.cosine_similarity(gold_mean.to(torch.float32), prompt_mean.to(torch.float32), dim=0)
                return similarity.item()
            elif prompt is None and gold_prompt is None:
                return 1
            else:
                return -1

        # -1 ~ 1
        scores = [
            get_similarity_score(positive_prompt, self.gold_positive_prompt, self.gold_positive_embedding_mean),
            get_similarity_score(negative_prompt, self.gold_negative_prompt, self.gold_negative_embedding_mean),
        ]

        # 0 ~ 1
        scores = [
            (cos_score + 1) / 2
            for cos_score in scores
        ]

        score = statistics.harmonic_mean(scores)
        submission: DiffuseGameSubmission = {
            "status": status,
            "account_url": status['account']['url'],
            "acct_as_mention": "@" + status['account']['acct'],
            "display_name": status['account']['display_name'],
            "positive": positive_prompt,
            "negative": negative_prompt,
            "score": score
        }

        self.submissions[status['account']['url']] = submission

        logging.info(f'game submission added by {status["account"]["acct"]}')


class DiffuseGameHandler(BotRequestHandler):
    class RequestType(enum.Enum):
        NewGame = 1
        AnswerSubmission = 2

    def __init__(self,
                 pipe: diffusers.pipelines.StableDiffusionPipeline,
                 tag_name: str = 'diffuse_game',
                 response_duration_sec: float = 60 * 1,
                 messages: DiffuseGameMessages = None
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

        self.current_game: Optional[DiffusionGameStatus] = None
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

        scores: List[(str, DiffuseGameSubmission)] = \
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
        reply_type: DiffuseGameHandler.RequestType = ctx.get_payload(type(self), 'req_type')

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

        current_game_status = ctx.mastodon.status_post(
            self.messages['new_game_start_announce'],
            media_ids=media_ids,
            visibility='unlisted',
            sensitive=True
        )

        self.current_game = DiffusionGameStatus(
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
                     + '\n\n' + self.current_game.status['url'])

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
