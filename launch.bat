cd /D "%~dp0"

chcp 65001

condaenv\bin\python -m diffusers_mastodon_bot.main

pause