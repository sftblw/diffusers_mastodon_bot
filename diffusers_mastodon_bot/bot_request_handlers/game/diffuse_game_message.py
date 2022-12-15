from typing_extensions import TypedDict
from typing import *


class DiffuseGameMessages(TypedDict):
    new_game_already_exists: str
    new_game_should_be_direct: str
    new_game_prompt_is_missing: str

    new_game_generation_in_progress: str

    new_game_start_announce: str
    new_game_start_success: str

    answer_submission_game_does_not_exist: str
    answer_submission_is_done_by_questioner: str

    answer_submission_no_chances_left: str
    answer_submission_left_chance_many: str
    answer_submission_left_chance_last: str
    answer_submission_left_chance_none: str
    answer_submission_perfect: str

    answer_submission_was_by_cw: str

    game_no_player: str
    game_no_player_cw: str
    game_end: str
    game_winner: str
    game_early_end: str

    question_by: str
    gold_positive_prompt: str
    gold_negative_prompt: str

_default_message = {
    "new_game_already_exists":
        "A diffusion game is already in progress.",
    "new_game_should_be_direct":
        "To create new game, send me # diffuse_game with \"direct\" message (aka private).",
    "new_game_prompt_is_missing":
        "Your prompt is missing! Please provide me # diffuse_me with prompt to generate an image, with DM.",
    "new_game_generation_in_progress":
        "creating new diffusion game...",
    "new_game_start_announce":
        "Diffusion guessing game is started! guess the prompt and reply to this message! \n\n#bot_message",
    "new_game_start_success":
        "ok! new game is generated! (You are not allowed to submit guesses, though!)",

    "answer_submission_game_does_not_exist":
        "This game does not exist anymore!",
    "answer_submission_is_done_by_questioner":
        "You are not allowed to answer to your question. Anyway You know it, isn't it?",

    "answer_submission_no_chances_left":
        "Oh, You used all your chance!",
    "answer_submission_left_chance_many":
        "Your score is {score}!\n\nYou have {chance_count} chances.",
    "answer_submission_left_chance_last":
        "Your score is {score}.\n\nYou have only one chance left!",
    "answer_submission_left_chance_none":
        "Your score is {score}.\n\nThat was last chance!",
    "answer_submission_perfect":
        "Your score is {score}, that's above {score_early_end_condition}! You are the finisher!",

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
    "game_early_end":
        "The game ended early with the correct answer!",

    "question_by": "This question was prompted by {account}!",
    "gold_positive_prompt": "prompt was:\n{prompt}",
    "gold_negative_prompt": "negative prompt was:\n{prompt}"
}

def diffusion_game_message_defaults(message: Optional[Dict[str, str]]) -> DiffuseGameMessages:
    new_message = _default_message.copy()

    if message is None:
        return new_message

    for key, value in message.items():
        new_message[key] = value

    return new_message


