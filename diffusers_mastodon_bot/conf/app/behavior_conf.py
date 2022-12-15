from dataclasses import dataclass


@dataclass
class BehaviorConf:
    default_visibility: str = 'unlisted'

    save_image: bool = True

    output_save_path: str = './diffused_results'

    save_args: bool = True
    save_args_text: bool = False

    delete_processing_message: bool = True
    tag_behind_on_image_post: bool = True
