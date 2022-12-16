import logging
from dataclasses import dataclass
from typing import Tuple


logger = logging.getLogger(__name__)


@dataclass
class ImageGenConf:
    image_count: int = 8
    max_image_count: int = 16
    image_tile_xy: Tuple[int, int] = 2, 2
    image_tile_auto_expand: bool = True
    image_max_attachment_count: int = 4
    max_batch_process: int = 1

    detect_nsfw: bool = True
    no_image_on_any_nsfw: bool = True

    def __post_init__(self):
        self.validate_prop()

    def validate_prop(self):
        if isinstance(self.image_tile_xy, list):
            self.image_tile_xy = tuple(self.image_tile_xy)

        self.image_count = self.image_count \
            if 1 <= (self.image_count / (self.image_tile_xy[0] * self.image_tile_xy[1])) <= self.image_max_attachment_count \
            else 1
        self.max_image_count = self.max_image_count \
            if 1 <= (self.max_image_count / (self.image_tile_xy[0] * self.image_tile_xy[1])) <= self.image_max_attachment_count \
            else 1

        # fixed
        # if self.max_batch_process is not None and self.max_batch_process >= 2:
        #     logger.warn('due to negative prompt bug in batch, it is temporary disabled. changing batch to 1... '
        #                 'https://github.com/huggingface/diffusers/issues/779')
        #     self.max_batch_process = 1
