from dataclasses import dataclass, field
from typing import *


@dataclass
class EmbeddingsConf:
    load_embeddings: bool = True
    embeddings_path: str = 'embeddings'
    prefix: str = ''
