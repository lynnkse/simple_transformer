import os
import sys

src_path = os.path.expanduser('/home/lynnkse/simple_transformer')
sys.path.append(src_path)

from transformer.Transformers.transformer import SimpleTransformer

model = SimpleTransformer()

# Forward pass example
text = "hello world"
output = model(text)
print(output)