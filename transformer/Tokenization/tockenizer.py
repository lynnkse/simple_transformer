# tokenizer.py
def tokenize(text):
    return text.split(' ')

def map_token_to_id(token, token_to_id_map):
    return token_to_id_map.get(token, 0)  # Return 0 for unknown tokens

token_to_id_map = {'hello': 1, 'world': 2, 'what': 3, 'is': 4, 'it': 5, 'this': 6, 'a': 7, 'test': 8}
