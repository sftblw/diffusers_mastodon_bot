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
    def __init__(self, mastodon_client, diffusers_pipeline, mention_to, tag_name='diffuse_me',
                 default_visibility='unlisted', output_save_path='./diffused_results',
                 toot_listen_start: Union[str, None]=None, toot_listen_end: Union[str, None]=None):
        self.mastodon: Mastodon = mastodon_client
        self.mention_to = mention_to
        self.tag_name = tag_name
        self.diffusers_pipeline = diffusers_pipeline
        self.output_save_path = output_save_path

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
        atexit.register(exit_toot)

        print('listening')
        self.mastodon.status_post(self.toot_listen_start, visibility=default_visibility)

    def on_update(self, status: Dict[str, any]):
        super().on_update(status)

        if status["visibility"] == 'public':
            return

        # [{'name': 'testasdf', 'url': 'https://don.naru.cafe/tags/testasdf'}]
        tags_list: List[Dict[str, any]] = status['tags']
        if self.tag_name not in map(lambda d: d['name'], tags_list):
            return

        # [{'id': 108719481602416740, 'username': 'sftblw', 'url': 'https://don.naru.cafe/@sftblw', 'acct': 'sftblw'}]
        mention_list: List[Dict[str, any]] = status["mentions"]
        if self.mention_to not in map(lambda d: d['acct'], mention_list):
            return

        reply_visibility = status['visibility']
        if reply_visibility == 'public':
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
        with autocast("cuda"):
            image: Image = self.diffusers_pipeline(content_txt)["sample"][0]

            end_time = time.time()

            image.save(image_filename, "PNG")
            Path(text_filename).write_text(content_txt)

        self.mastodon.status_delete(in_progress_status['id'])
        image_posted = self.mastodon.media_post(image_filename, 'image/png')

        time_took = end_time - start_time
        time_took = int(time_took * 1000) / 1000

        reply_message = f'took: {time_took}s' + '\n\n' + f'prompt: \n{content_txt}'
        if len(reply_message) > 500:
            reply_message = reply_message[0:480] + '...'

        self.mastodon.status_reply(status, reply_message, media_ids=[image_posted['id']], visibility=reply_visibility)