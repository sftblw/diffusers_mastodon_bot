import logging
import statistics
from typing import *

import torch
import transformers

from diffusers_mastodon_bot.bot_request_handlers.diffusion_runner import DiffusionRunner
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_submission import DiffuseGameSubmission


class DiffuseGameStatus:
    def __init__(self,
                 tokenizer: transformers.CLIPTokenizer,
                 text_encoder: transformers.CLIPTextModel,
                 status: Dict[str, Any],
                 submitter_url: str,
                 submitter_acct: str,
                 positive_prompt: Optional[str],
                 negative_prompt: Optional[str],
                 left_chance = 3
                 ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # status or None
        self.status: Optional[Dict[str, Any]] = status
        self.submitter_url = submitter_url
        self.submitter_acct = submitter_acct

        self.gold_positive_prompt: Optional[str] = positive_prompt
        self.gold_negative_prompt: Optional[str] = negative_prompt

        self.left_chance = left_chance

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
        prompt_tensor: torch.Tensor = DiffusionRunner.embed_prompt(
            prompt=prompt, tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        prompt_tensor = prompt_tensor.to('cpu')
        return prompt_tensor

    def set_submission(self,
                       status: Dict[str, Any],
                       positive_prompt: Optional[str],
                       negative_prompt: Optional[str]
                       ):

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
        score_positive = get_similarity_score(positive_prompt, self.gold_positive_prompt, self.gold_positive_embedding_mean)
        score_negative = get_similarity_score(negative_prompt, self.gold_negative_prompt, self.gold_negative_embedding_mean)

        # 0 ~ 1
        score_positive = (score_positive + 1) / 2
        score_negative = (score_negative + 1) / 2

        scores = [
            score_positive,
            score_negative
        ]

        score = statistics.harmonic_mean(scores)

        submission: DiffuseGameSubmission = {
            "status": status,
            "account_url": status['account']['url'],
            "acct_as_mention": "@" + status['account']['acct'],
            "display_name": status['account']['display_name'],
            "positive": positive_prompt,
            "negative": negative_prompt,
            "score": score,
            "score_positive": score_positive,
            "score_negative": score_negative
        }

        self.submissions[status['account']['url']] = submission

        logging.info(f'game submission added by {status["account"]["acct"]} - positive score {score_positive}, negative score {score_negative}')
