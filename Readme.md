# duffusers_mastodon_bot

a quick and dirty bot, running stable diffuser, via huggingface diffusers

## prepare & run

- virtualenv
- install pytorch, via pip, enabling nvidia
- `pip install -r requirements.txt`
- `huggingface-cli login` (see hf diffusers)
- create app from account setting, fill these text files
  - `config/access_token.txt`
  - `config/endpoint_url.txt` ex) `https://mastodon.social
  - optional
    - `config/toot_listen_start.txt`: toot content on listening start
    - `config/toot_listen_end.txt`: toot content on listening end (exit)
    - `config/proc_kwargs.json`: additional parameters dictionary,
      for [\_\_call__](https://github.com/huggingface/diffusers/blob/v0.2.4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L39-L51)
- `python -m diffusers_mastodon_bot.main`

## use

mention to target acct user with tag, hardcoded are:

- user itself
- #diffuse_me

```text
@diffuse_bot@example.com #diffuse_me prompt text shown in dark monitor in parking lot
```

## config examples

### `config/proc_kwargs.json`

bug?: https://github.com/huggingface/diffusers/issues/255

```json
{
  "width": 640,
  "height": 512,
  "num_inference_steps": 70
}
```

### `config/app_stream_kwargs.json`

```json
{
  "image_count": 4,
  "max_image_count": 16,
  "image_tile_xy": [2, 2],
  "max_batch_process": 2,
  "delete_processing_message": false,
  "toot_on_start_end": false
}
```