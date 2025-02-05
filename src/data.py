import torch
import os
from datasets import Dataset

def _tokenize(dictionary, ds=None, type_ds=None, path=None, limit_line=None):
    nb_tokens_in_dictionary = len(dictionary)
    # load document to tokenize
    if ds is not None:
        document = ds[type_ds]["text"]
    elif path is not None:
        document = Dataset.from_text(path_or_paths=path)["text"]
    else:
        raise("Don't find documnet to tokenize")

    # Count nb of tokens in text and update the dictionary
    for i, line in enumerate(document):
        if i == limit_line:
            break
        tokens = line.split() + ["<eos>"]
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = nb_tokens_in_dictionary
                nb_tokens_in_dictionary += 1

    # Assign to each token its identifier
    ids = []
    for i, line in enumerate(document):
        if i == limit_line:
            break
        i += 1
        tokens = line.split() + ["<eos>"]
        for token in tokens:
            ids.append(dictionary[token])
    ids = torch.LongTensor(ids)
    return ids


class Corpus:
    def __init__(self, ds=None, path=None):
        self._dictionary = {}
        self.train = _tokenize(
            dictionary=self._dictionary, ds=ds, type_ds="train", path=os.path.join(path, "train.txt")
        )
        self.valid = _tokenize(
            dictionary=self._dictionary, ds=ds, type_ds="validation", path=os.path.join(path, "valid.txt")
        )
        self.test = _tokenize(
            dictionary=self._dictionary, ds=ds, type_ds="test", path=os.path.join(path, "test.txt")
        )

    @property
    def vocab_size(self):
        return len(self._dictionary)