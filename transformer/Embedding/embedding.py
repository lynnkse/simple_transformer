# embedding.py
import torch
import torch.nn as nn

d_model = 512
vocab_size = 1000  # For demonstration
embedding_layer = nn.Embedding(vocab_size, d_model)
