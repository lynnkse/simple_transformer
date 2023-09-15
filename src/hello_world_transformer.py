import torch
import torch.nn as nn

# Mock tokenizer
def mock_tokenizer(sentence):
    vocab = {'hello': 1, 'world': 2, 'what': 3, 'is': 4, 'it': 5, '?': 6, 'this': 7, 'a': 8, 'test': 9, '!': 10}
    return [vocab.get(word, 0) for word in sentence.lower().split()]

# Hyperparameters
d_model = 512
nhead = 8
num_layers = 3
dim_feedforward = 2048

# Source and target sequences
src_sentence = "hello world what is it?"
tgt_sentence = "This is a test!"
src = torch.LongTensor([mock_tokenizer(src_sentence)])
tgt = torch.LongTensor([mock_tokenizer(tgt_sentence)])

# Embedding layers
input_embedding = nn.Embedding(1000, d_model)
output_embedding = nn.Embedding(1000, d_model)

# Transformer model
model = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward)

# Forward pass
src = input_embedding(src)
tgt = output_embedding(tgt)
out = model(src, tgt)

print(out)
