import re
from typing import List
from src.nlp.tokenizer import Tokenizer
from src.nlp.sentencepiece.processor import Processor

class LlamaTokenizer(Tokenizer):
    def __init__(self, tokenizer_model_path: str):
        self.sp_model = Processor(tokenizer_model_path)

    def encode(self, text: str):
        """Encodes text into list of integers.
        
        Args:
            text (str): Text to encode.
                
        Returns:
            List of integers representing encoded string.  
        """
        encoded = []
        # We manually replace <s> and </s> with the correct bos/eos ids
        match = re.search('<s>|</s>', text)
        while match is not None:
            if match.group(0) == '<s>':
                encoded.append(self.sp_model.bos_id())
            else:
                encoded.append(self.sp_model.eos_id())
            encoded += self.sp_model.encode(text[:match.span()[0]])
            text = text[match.span()[1]:]
            match = re.search('<s>|</s>', text)
        encoded += self.sp_model.encode(text)
        
        return encoded

    def decode(self, indices: List[int], remove_leading_space: bool=False):
        return self.sp_model.decode(indices, remove_dummy_prefix=remove_leading_space)