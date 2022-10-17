from typing import *


class DiffuseGameSubmission(TypedDict):
    status: Dict[str, Any]
    account_url: str
    acct_as_mention: str
    display_name: str
    positive: Optional[str]
    negative: Optional[str]
    score: float
    score_positive: float
    score_negative: float
    left_chance: int
