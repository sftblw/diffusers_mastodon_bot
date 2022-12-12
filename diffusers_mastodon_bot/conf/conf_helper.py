from pathlib import Path
from omegaconf.omegaconf import OmegaConf


def load_structured_conf_yaml(conf_type: type, file_path: str):
    conf = OmegaConf.structured(conf_type)
    if Path(file_path).exists():
        conf = OmegaConf.unsafe_merge(conf, OmegaConf.load(file_path))
    return conf