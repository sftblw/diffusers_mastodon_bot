import abc
from typing import *

from .bot_request_context import BotRequestContext
from .proc_args_context import ProcArgsContext


class BotRequestHandler(abc.ABC):
    """
    simple Request Handling pipeline implementation. first comes first.
    """
    @abc.abstractmethod
    def is_eligible_for(self, ctx: BotRequestContext) -> bool:
        pass

    @abc.abstractmethod
    def respond_to(self, ctx: BotRequestContext, args_ctx: ProcArgsContext) -> bool:
        pass
