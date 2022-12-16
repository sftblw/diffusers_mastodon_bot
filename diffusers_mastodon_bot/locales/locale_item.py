import logging
from typing import Optional, Callable


logger = logging.getLogger(__name__)


class LocaleItem:
    def __init__(self, key, message_function: Optional[Callable]):
        self.key = key
        self.message_function = message_function

    def __call__(self, **kwargs):
        errors = []
        kwargs = dict(kwargs)

        msg = ''
        if self.message_function is not None:
            try:
                msg = self.message_function(kwargs, errors)
            except Exception as ex:
                logger.error(f"can't load message for {self.key}: {ex}")
        else:
            logger.info(f"no message for {self.key}, returning empty one")
            msg = ''

        if len(errors) > 0:
            logger.error(f'errors on formatting locale message {self.key}: {errors}')

        return msg
