from typing import *

import tokenizers
import torch
from transformers import CLIPTextModel, CLIPTokenizer


# Since text encoder does not support multiple tokens, this is a workaround.
class CustomTokenHelper:
    def __init__(self, prefix: str, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer):
        self.prefix = prefix
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.multiple_tokens: Dict[str, int] = {}

        self.added_custom_tokens = []

    def add_new_custom_token(self, new_token_name: str, new_embedding_tensor: torch.Tensor) -> str:
        """
        add custom token as a new special token.
        returns new name of the token(s), as some prefix can be placed above the input text
        """
        if len(self.prefix) >= 1:
            new_token_name = self.prefix + new_token_name

        embedding: torch.nn.Embedding = self.text_encoder.get_input_embeddings()  # type: ignore

        new_embedding_length = int(new_embedding_tensor.shape[0])

        if new_embedding_length == 1:
            # no touch
            self.tokenizer.add_tokens(tokenizers.AddedToken(new_token_name))
            embedding = self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            embedding.weight.data[-1] = new_embedding_tensor[0].to(embedding.weight.data.device)
        elif new_embedding_length >= 2:
            # replaces into {token}_{i}
            self.multiple_tokens[new_token_name] = new_embedding_length

            for i in range(new_embedding_length):
                self.tokenizer.add_tokens(tokenizers.AddedToken(f'{new_token_name}_{i}'))

            embedding = self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            embedding.weight.data[-new_embedding_length:] = new_embedding_tensor.to(embedding.weight.data.device)

        embedding.requires_grad_(False)
        self.text_encoder.set_input_embeddings(embedding)

        self.added_custom_tokens.append(new_token_name)

        return new_token_name

    # awful efficiency, but for not touching library code.
    def apply_custom_multiple_tokens(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None

        for key, value in self.multiple_tokens.items():
            target_text = ''.join(map(lambda i: f'{key}_{i}', range(value)))
            text = text.replace(key, target_text)

        return text
