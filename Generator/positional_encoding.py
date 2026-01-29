import math
import torch

def even_position(p, i, dim):
    return math.sin(p / (10000 ** ((2 * i) / dim)))

def odd_position(p, i, dim):
    return math.cos(p / (10000 ** ((2 * i) / dim)))

def positional_encoding(tokens_len, embed_dim):
    positional_encodings = []
    for p in range(tokens_len):
        token_position = []
        for i in range(embed_dim):
            if i % 2 == 0:
                token_position.append(even_position(p, i, embed_dim))
            else:
                token_position.append(odd_position(p, i, embed_dim))
        positional_encodings.append(torch.tensor(token_position))
    return torch.stack(positional_encodings)
