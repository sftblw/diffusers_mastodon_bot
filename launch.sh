#!/bin/bash

export HF_HOME="`pwd`/hf_cache"
mkdir -p $HF_HOME

./condaenv/bin/python -m diffusers_mastodon_bot.main