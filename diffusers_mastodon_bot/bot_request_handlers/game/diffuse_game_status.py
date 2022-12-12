import logging
import statistics
from typing import *

import torch
import transformers

from diffusers_mastodon_bot.bot_request_handlers.diffusion_runner import DiffusionRunner
from diffusers_mastodon_bot.bot_request_handlers.game.diffuse_game_submission import DiffuseGameSubmission


logger = logging.getLogger(__name__)


class DiffuseGameStatus:
    def __init__(self,
                 status: Dict[str, Any],
                 questioner_url: str,
                 questioner_acct: str,
                 positive_prompt: Optional[str],
                 negative_prompt: Optional[str],
                 calc_weighted_embeddings: Callable[[str, str], Tuple[torch.Tensor, torch.Tensor]],
                 initial_chance=5
                 ):
        # status or None
        self.status: Optional[Dict[str, Any]] = status
        self.eligible_status_ids_for_reply: Set[Dict[str, Any]] = set()

        self.questioner_url = questioner_url
        self.questioner_acct = questioner_acct

        self.gold_positive_prompt: Optional[str] = positive_prompt
        self.gold_negative_prompt: Optional[str] = negative_prompt
        self.calc_weighted_embeddings = calc_weighted_embeddings

        positive_embedding, negative_embedding = self.prompt_as_embedding(positive_prompt, negative_prompt)

        self.initial_chance = initial_chance

        # tokenizer -> embedding -> last layer
        # (utilizes fn from community pipeline)
        # torch.Size([1, 77, 768])
        self.gold_positive_embedding: Optional[torch.Tensor] = positive_embedding
        self.gold_positive_embedding_mean = \
            self.gold_positive_embedding[0].mean(dim=0) if self.gold_positive_embedding is not None else None
        # torch.Size([768])

        self.gold_negative_embedding: Optional[torch.Tensor] = negative_embedding
        self.gold_negative_embedding_mean = \
            self.gold_negative_embedding[0].mean(dim=0) if self.gold_negative_embedding is not None else None
        # torch.Size([768])

        # for multiple submission, dictionary with acct is used.
        self.submissions: Dict[str, DiffuseGameSubmission] = {}

    def prompt_as_embedding(self, positive: str, negative: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        positive_tensor, negative_tensor = self.calc_weighted_embeddings(positive, negative)
        if positive_tensor is not None:
            positive_tensor = positive_tensor.to('cpu')

        if negative_tensor is not None:
            negative_tensor = negative_tensor.to('cpu')

        return positive_tensor, negative_tensor

    def set_submission(self,
                       status: Dict[str, Any],
                       positive_prompt: Optional[str],
                       negative_prompt: Optional[str],
                       include_negative_on_final_score: bool = False
                       ) -> DiffuseGameSubmission:
        """
        :param status: submission to process
        :param positive_prompt: positive answer submission, preprocessed
        :param negative_prompt: negative answer submission, preprocessed
        :param include_negative_on_final_score: Most users do not count on negative answer. do not include them on final answer.
        :return: new submission created
        """

        def get_similarity_score(embedding: Optional[torch.Tensor], gold_prompt: Optional[str], gold_mean: torch.Tensor):
            if embedding is not None and gold_prompt is not None:
                # torch.Size([1, 77, 768]) -> (indexing 0) -> (77, 768)
                prompt_embedding: torch.Tensor = embedding[0]
                prompt_mean = prompt_embedding.mean(dim=0)
                # [768]
                similarity: torch.Tensor = \
                    torch.cosine_similarity(gold_mean.to(torch.float32), prompt_mean.to(torch.float32), dim=0)
                return similarity.item()
            elif embedding is None and gold_prompt is None:
                return 1
            else:
                return 0

        positive_embedding, negative_embedding = self.prompt_as_embedding(positive_prompt, negative_prompt)

        # -1 ~ 1
        score_positive = get_similarity_score(positive_embedding, self.gold_positive_prompt, self.gold_positive_embedding_mean)
        score_negative = get_similarity_score(negative_embedding, self.gold_negative_prompt, self.gold_negative_embedding_mean)

        # 0 ~ 1
        score_positive = (score_positive + 1) / 2
        score_negative = (score_negative + 1) / 2

        if include_negative_on_final_score:
            scores = [
                score_positive,
                score_negative
            ]

            score = statistics.harmonic_mean(scores)
        else:
            score = score_positive

        submitter_url = status['account']['url']
        previous_submission: Optional[DiffuseGameSubmission] = (
            self.submissions[submitter_url] if submitter_url in self.submissions.keys() else None
        )

        submission: DiffuseGameSubmission = {
            "status": status,
            "account_url": status['account']['url'],
            "acct_as_mention": "@" + status['account']['acct'],
            "display_name": status['account']['display_name'],
            "positive": positive_prompt,
            "negative": negative_prompt,
            "score": score,
            "score_positive": score_positive,
            "score_negative": score_negative,
            "left_chance": self.initial_chance - 1 if previous_submission is None else previous_submission['left_chance'] - 1
        }

        if previous_submission is not None and previous_submission['score'] > submission['score']:
            self.submissions[submitter_url]['left_chance'] = submission['left_chance']
        else:
            self.submissions[submitter_url] = submission

        logger.info(f'game submission added by {status["account"]["acct"]} - positive score {score_positive}, negative score {score_negative}')

        return submission

    def left_chance_for(self, submitter_url: str) -> int:

        if submitter_url not in self.submissions.keys():
            return self.initial_chance

        return self.submissions[submitter_url]['left_chance']

    def register_status_as_eligible_for_reply(self, status: Dict[str, Any]):
        self.eligible_status_ids_for_reply.add(status['id'])
