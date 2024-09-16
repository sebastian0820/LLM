import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab                              #A
        self.int_to_str = {i:s for s,i in vocab.items()}     #B
    def encode(self, text):                                      #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # print(preprocessed)
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):                                       #D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)          #E
        return text