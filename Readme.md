# duffusers_mastodon_bot

a quick and dirty bot, running stable diffuser, via huggingface diffusers

## prepare & run

- virtualenv (or conda, I recommend [micromamba](https://github.com/mamba-org/mamba))
- install pytorch, via pip, enabling nvidia
- `pip install -r requirements.txt`
  - (You can remove optional dependencies from there)
- `huggingface-cli login` (see hf diffusers)
- create app from account setting, fill these text files
  - minimal: `config/instance.yaml`
    - see `config_example`
    - missing config becomes default, see [`diffusers_mastodon_bot/conf`](./diffusers_mastodon_bot/conf)
- `python -m diffusers_mastodon_bot.main`
- optional things for performance:
  - install `accelerate`
  - install `xformers`
    - On linux with conda(or mamba), it's easy to install
    - On Windows, [see this](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2103)   

## features

- image generation: mentioning the bot with `#diffuse_me` and prompt
  - If you are the bot, You can do it without mention
- image2image: mention with `#diffuse_it` and one attachment image
- image generation game: mentioning the bot in DM with `#diffuse_game` and prompt

```text
@bot@example.com 

#diffuse_me 

args.orientation landscape
args.image_count 16
args.guidance_scale 30
args.num_inference_steps 70

suzuran from arknights at cozy cafe with tea.
extremely cute, round face, big fox ears directing side,
cyan hairband, bright gold yellow hair with thin twintail
as round shape braided, bangs, round gentle emerald eyes,
girlish comfort dress, 1girl, urban fantasy, sci-fi,
comfort art style, ultra detail, highres

sep.negative

high contrast, trecen school suite, uma musume
```

## config examples

see `config_example`. copy-paste it to `config` and modify from there.

## misc

This software contains source from `huggingface/diffusers`, which is under Apache License 2.