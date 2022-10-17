from typing_extensions import TypedDict
from typing import *

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

def diffusion_game_message_defaults(message: Dict[str, str]):
    pass

