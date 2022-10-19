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
    - see `config_example`
- `python -m diffusers_mastodon_bot.main`

## features

- image generation: mentioning the bot with `#diffuse_me` and prompt
  - If you are the bot, You can do it without mention
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