# Simple Transformer Implementation
Author: Anton Gulyaev  
Contact: lynnkse@gmail.com  

## Overview
This repository provides a simplified implementation of a Transformer, a type of neural network architecture that is the backbone for models like GPT-3, BERT, etc. It demonstrates fundamental aspects such as tokenization, embedding, encoder-decoder architecture, and forward propagation.

## High-Level Design
The high-level design of the Transformer is as follows:  
1. **Source Sequence (Input Sentence)**  
   Example: "hello world what is it? This is a test!"
2. **Pre-Processing (Tokenization)**  
   Example: `["hello", "world", "what", "is", "it", "?", "This", "is", "a", "test", "!"]`
3. **Positionally Encoded Vectors (Embedding Layer)**  
   Example: Tensors corresponding to each token with positional encoding.
4. **Encoder**  
   Processes the embedded sequence and produces encoder hidden states.
5. **Decoder**  
   Processes the encoder hidden states and produces the target sequence.
6. **De-Tokenization**  
   Converts the target sequence back to human-readable text.

## Code Snippets
Below are simplified representations to illustrate the components and flow.

### SimpleTransformer Class
```python
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, text):
        tokens = tokenize(text)  # Tokenization
        token_ids = [map_token_to_id(token, token_to_id_map) for token in tokens]  # Mapping tokens to IDs
        token_tensor = torch.LongTensor(token_ids).unsqueeze(0)  # Conversion to Tensor
        
        embedded = embedding_layer(token_tensor)  # Embedding
        encoder_output = self.encoder(embedded)  # Encoding
        decoder_output = self.decoder(encoder_output)  # Decoding
        
        return decoder_output