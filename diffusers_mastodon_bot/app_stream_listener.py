from typing import *

import unicodedata
import atexit
import re
import time
from pathlib import Path
from datetime import datetime

import mastodon
from PIL import Image
from mastodon import Mastodon
from torch import autocast

from bs4 import BeautifulSoup


def rip_out_html(text):
    return BeautifulSoup(text, features="html.parser").get_text()


class AppStreamListener(mastodon.StreamListener):
    def __init__(self, mastodon_client, diffusers_pipeline, mention_to_url, tag_name='diffuse_me',
                 default_visibility='unlisted', output_save_path='./diffused_results',
                 toot_listen_start: Union[str, None] = None, toot_listen_end: Union[str, None] = None,
                 delete_processing_message=False,
                 device: str = 'cuda',
                 proc_kwargs: Union[None, Dict[str, any]] = None):
        self.mastodon: Mastodon = mastodon_client
        self.mention_to_url = mention_to_url
        self.tag_name = tag_name
        self.diffusers_pipeline = diffusers_pipeline
        self.output_save_path = output_save_path

        self.delete_processing_message = delete_processing_message

        self.proc_kwargs = proc_kwargs
        self.device = device

        self.strippers = [
            re.compile('@[a-zA-Z0-9._-]+'),
            re.compile('#[a-zA-Z0-9_-]+'),
            re.compile('[ \r\n\t]+'),
        ]

        if not Path(self.output_save_path).is_dir():
            Path(self.output_save_path).mkdir(parents=True, exist_ok=True)

        self.toot_listen_start = toot_listen_start
        self.toot_listen_end = toot_listen_end

        if self.toot_listen_start is None:
            self.toot_listen_start = f'listening (diffusers_mastodon_bot)\n\nsend me a prompt with hashtag # {tag_name}'

        if self.toot_listen_end is None:
            self.toot_listen_end = f'exiting (diffusers_mastodon_bot)\n\ndo not send me anymore'

        def exit_toot():
            self.mastodon.status_post(self.toot_listen_end, visibility=default_visibility)
            pass

        atexit.register(exit_toot)

        print('listening')
        self.mastodon.status_post(self.toot_listen_start, visibility=default_visibility)

    def status_contains_target_tag(self, status):
        # [{'name': 'testasdf', 'url': 'https://don.naru.cafe/tags/testasdf'}]
        tags_list: List[Dict[str, any]] = status['tags']
        if self.tag_name not in map(lambda tag: tag['name'], tags_list):
            return False
        return True

    def on_notification(self, notification):
        noti_type = notification['type']
        if noti_type != 'mention':
            return

        status = notification['status']

        if not self.status_contains_target_tag(status):
            return

        # [{'id': 108719481602416740, 'username': 'sftblw', 'url': 'https://don.naru.cafe/@sftblw', 'acct': 'sftblw'}]
        mention_list: List[Dict[str, any]] = status["mentions"]
        if self.mention_to_url not in map(lambda account: account['url'], mention_list):
            return

        self.respond_to(status)

    # self response, without notification
    def on_update(self, status: Dict[str, any]):
        super().on_update(status)

        account = status['account']
        if account['url'] != self.mention_to_url:
            return

        if not self.status_contains_target_tag(status):
            return

        self.respond_to(status)

    def respond_to(self, status):
        reply_visibility = status['visibility']
        if reply_visibility == 'public' or reply_visibility == 'direct':
            reply_visibility = 'unlisted'

        content_txt = rip_out_html(status['content'])
        print(f'text : {content_txt}')

        for stripper in self.strippers:
            content_txt = stripper.sub(' ', content_txt).strip()

        content_txt = unicodedata.normalize('NFC', content_txt)

        print(f'text (strip out) : {content_txt}')

        in_progress_status = self.mastodon.status_reply(status, 'processing...', visibility=reply_visibility)

        print('starting')

        start_time = time.time()

        filename_root = datetime.now().strftime('%Y-%m-%d') + f'_{str(start_time)}'

        image_filename = self.output_save_path + '/' + filename_root + '.png'
        text_filename = self.output_save_path + '/' + filename_root + '.txt'

        proc_kwargs = self.proc_kwargs if self.proc_kwargs is not None else {}
        with autocast(self.device):
            pipe_result = self.diffusers_pipeline(content_txt, **proc_kwargs)
            image: Image = pipe_result["sample"][0]
            nsfw_content_detected: bool = pipe_result['nsfw_content_detected'][0]

            end_time = time.time()

            image.save(image_filename, "PNG")
            Path(text_filename).write_text(content_txt)

        if self.delete_processing_message:
            self.mastodon.status_delete(in_progress_status['id'])

        image_posted = self.mastodon.media_post(image_filename, 'image/png')

        time_took = end_time - start_time
        time_took = int(time_took * 1000) / 1000

        reply_message = f'took: {time_took}s'

        if self.proc_kwargs is not None and 'num_inference_steps' in self.proc_kwargs:
            reply_message += '\n\n' + f'inference steps: {self.proc_kwargs["num_inference_steps"]}'

        if nsfw_content_detected:
            reply_message += '\n\n' + 'nsfw content detected, result will be a empty image'

        reply_message += '\n\n' + f'prompt: \n{content_txt}'

        if len(reply_message) > 500:
            reply_message = reply_message[0:480] + '...'

        reply_target_status = status if self.delete_processing_message else in_progress_status
        self.mastodon.status_reply(reply_target_status, reply_message, media_ids=[image_posted['id']],
                                   visibility=reply_visibility,
                                   sensitive=True
                                   )
