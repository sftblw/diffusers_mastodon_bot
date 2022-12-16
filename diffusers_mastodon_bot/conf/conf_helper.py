from pathlib import Path
from typing import Optional

from omegaconf import SCMode
from omegaconf.omegaconf import OmegaConf


def load_structured_conf_yaml(
        conf_type: type,
        file_path: str,
        instanciate: bool = True
):
    # if instanciate:
    conf = OmegaConf.structured(conf_type, )
    if Path(file_path).exists():
        conf = OmegaConf.merge(conf, OmegaConf.load(file_path))

    if instanciate:
        conf = OmegaConf.to_container(conf, structured_config_mode=SCMode.INSTANTIATE)

    return conf
