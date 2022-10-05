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

import requests
from io import BytesIO
from PIL import Image

def rip_out_html(text: str):
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ").replace('</p>', '</p> ')
    return BeautifulSoup(text, features="html.parser").get_text()


class AppStreamListener(mastodon.StreamListener):
    def __init__(self, mastodon_client, diffusers_pipeline, mention_to_url,
                 tag_name='diffuse_me',
                 default_visibility='unlisted', output_save_path='./diffused_results',
                 toot_listen_start: Union[str, None] = None, toot_listen_end: Union[str, None] = None,
                 delete_processing_message=False,
                 image_count=1,
                 device: str = 'cuda',
                 toot_on_start_end=True,
                 no_image_on_any_nsfw=True,
                 proc_kwargs: Union[None, Dict[str, any]] = None):
        self.mastodon: Mastodon = mastodon_client
        self.mention_to_url = mention_to_url
        self.tag_name = tag_name
        self.diffusers_pipeline = diffusers_pipeline
        self.output_save_path = output_save_path
        self.no_image_on_any_nsfw = no_image_on_any_nsfw

        self.delete_processing_message = delete_processing_message
        self.image_count = image_count if 1 <= image_count <= 4 else 1

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
            if toot_on_start_end:
                self.mastodon.status_post(self.toot_listen_end, visibility=default_visibility)
            pass

        atexit.register(exit_toot)

        print('listening')
        if toot_on_start_end:
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

        try:
            self.respond_to(status)
        except Exception as ex:
            print(f'error on notification respond: {str(ex)}')
            pass

    # self response, without notification
    def on_update(self, status: Dict[str, any]):
        super().on_update(status)

        account = status['account']
        if account['url'] != self.mention_to_url:
            return

        if not self.status_contains_target_tag(status):
            return

        try:
            self.respond_to(status)
        except Exception as ex:
            print(f'error on self status respond: {str(ex)}')
            pass

    def respond_to(self, status):
        reply_visibility = status['visibility']
        if reply_visibility == 'public' or reply_visibility == 'direct':
            reply_visibility = 'unlisted'

        print(f'html : {status["content"]}')
        content_txt = rip_out_html(status['content'])
        print(f'text : {content_txt}')

        for stripper in self.strippers:
            content_txt = stripper.sub(' ', content_txt).strip()

        content_txt = unicodedata.normalize('NFC', content_txt)

        print(f'text (strip out) : {content_txt}')

        in_progress_status = self.mastodon.status_reply(status, 'processing...', visibility=reply_visibility)

        print('starting')

        proc_kwargs = self.proc_kwargs if self.proc_kwargs is not None else {}
        proc_kwargs = proc_kwargs.copy()

        if 'width' not in proc_kwargs: proc_kwargs['width'] = 512
        if 'height' not in proc_kwargs: proc_kwargs['height'] = 512

        # param parsing
        tokens = [tok.strip() for tok in content_txt.split(' ')]

        before_args_name = None
        new_content_txt = ''

        for tok in tokens:
            if tok.startswith('args.'):
                before_args_name = tok[5:]
                continue

            if before_args_name is not None:
                args_value = tok

                if before_args_name in ['num_inference_steps', 'guidance_scale', 'orientation']:

                    if before_args_name == 'orientation':
                        if (
                                args_value == 'landscape' and proc_kwargs['width'] < proc_kwargs['height']
                                or args_value == 'portrait' and proc_kwargs['width'] > proc_kwargs['height']
                        ):
                            width_backup = proc_kwargs['width']
                            proc_kwargs['width'] = proc_kwargs['height']
                            proc_kwargs['height'] = width_backup

                    elif before_args_name in ['num_inference_steps']:
                        proc_kwargs[before_args_name] = min(int(args_value), 100)

                    else:
                        proc_kwargs[before_args_name] = min(float(args_value), 100.0)

                before_args_name = None
                continue

            new_content_txt += ' ' + tok

        content_txt = new_content_txt.strip()

        print(f'text (after argparse) : {content_txt}')

        # start
        start_time = time.time()
        time_took = 0

        filename_root = datetime.now().strftime('%Y-%m-%d') + f'_{str(start_time)}'

        generated_image_paths = []
        has_any_nsfw = False

        with autocast(self.device):
            for idx in range(self.image_count):
                image_filename = str(Path(self.output_save_path, filename_root + f'_{idx}' + '.png').resolve())
                text_filename = str(Path(self.output_save_path, filename_root + f'_{idx}' + '.txt').resolve())

                pipe_result = self.diffusers_pipeline(content_txt, **proc_kwargs)

                end_time = time.time()
                time_took += end_time - start_time

                image: Image = pipe_result.images[0]
                nsfw_content_detected: bool = pipe_result.nsfw_content_detected

                if nsfw_content_detected:
                    has_any_nsfw = True

                    if self.no_image_on_any_nsfw:
                        break

                image.save(image_filename, "PNG")
                Path(text_filename).write_text(content_txt)

                generated_image_paths.append(image_filename)

        if self.delete_processing_message:
            self.mastodon.status_delete(in_progress_status['id'])

        images_list_posted: List[Dict[str, any]] = []

        for image_path in generated_image_paths:
            try:
                upload_result = self.mastodon.media_post(image_path, 'image/png')
                images_list_posted.append(upload_result)
            except Exception as ex:
                print(f'error on image upload: {ex}')
                pass

        time_took = int(time_took * 1000) / 1000

        reply_message = f'took: {time_took}s'

        def detect_args_and_print(args_name):
            if proc_kwargs is not None and args_name in proc_kwargs:
                return '\n' + f'{args_name}: {proc_kwargs[args_name]}'
            else:
                return ''

        reply_message += detect_args_and_print('num_inference_steps')
        reply_message += detect_args_and_print('guidance_scale')

        if has_any_nsfw:
            reply_message += '\n\n' + 'nsfw content detected, some of result will be a empty image'

        reply_message += '\n\n' + f'prompt: \n{content_txt}'

        if len(reply_message) > 500:
            reply_message = reply_message[0:480] + '...'

        media_ids = [image_posted['id'] for image_posted in images_list_posted]
        if has_any_nsfw and self.no_image_on_any_nsfw:
            media_ids = None

        reply_target_status = status if self.delete_processing_message else in_progress_status
        self.mastodon.status_reply(reply_target_status, reply_message,
                                   media_ids=media_ids,
                                   visibility=reply_visibility,
                                   sensitive=True
                                   )
