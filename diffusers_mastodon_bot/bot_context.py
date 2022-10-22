from typing import *


class BotContext:
    def __init__(self,
                 bot_acct_url: str,
                 output_save_path: str,
                 max_batch_process: int,
                 delete_processing_message: bool,
                 no_image_on_any_nsfw: bool,
                 image_tile_xy: Tuple[int, int],
                 image_tile_auto_expand: bool,
                 image_max_attachment_count: int,
                 default_visibility: str,
                 device_name: str
                 ):
        self.bot_acct_url = bot_acct_url
        self.output_save_path = output_save_path
        self.max_batch_process =  max_batch_process
        self.delete_processing_message = delete_processing_message
        self.no_image_on_any_nsfw = no_image_on_any_nsfw
        self.image_tile_xy = image_tile_xy
        self.image_tile_auto_expand = image_tile_auto_expand
        self.image_max_attachment_count = image_max_attachment_count
        self.default_visibility = default_visibility
        self.device_name = device_name