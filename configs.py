# Define all the basic necessary things like batch size max_len paths etc. in a config class.

import tokenizers
import transformers

MAX_LEN = 128
# TOKENIZER = tokenizers.BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
