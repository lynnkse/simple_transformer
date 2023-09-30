# transformer.py
import torch
import torch.nn as nn
from ..Tokenization.tockenizer import tokenize, map_token_to_id, token_to_id_map
from ..Embedding.embedding import embedding_layer
from ..Embedding.positional_encodding import PositionalEncoding
from ..Encoding.encoder import Encoder
from ..Decoding.decoder import Decoder

class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.pos_encoder = PositionalEncoding(512) #d_model = 512, should be same as embedding in its package

    def forward(self, text):
        tokens = tokenize(text)
        token_ids = [map_token_to_id(token, token_to_id_map) for token in tokens]
        token_tensor = torch.LongTensor(token_ids).unsqueeze(0)  # [1, seq_length]

        embedded = embedding_layer(token_tensor)  # [1, seq_length, d_model]
        encoder_output = self.encoder(embedded)
        decoder_output = self.decoder(encoder_output)
        
        return decoder_output

# Create a SimpleTransformer instance

