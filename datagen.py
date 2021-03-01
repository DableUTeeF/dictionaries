from torch.utils.data import Dataset
from torchtext.vocab import Vocab, Counter
import os
import json
import torch


def generate_batch(batch):
    label = torch.tensor([entry[2].int() for entry in batch])
    text = [entry[0] for entry in batch]
    word = [entry[1] for entry in batch]
    text_offsets = [0] + [len(entry) for entry in text]
    word_offsets = [0] + [len(entry) for entry in word]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    text_offsets = torch.tensor(text_offsets[:-1]).cumsum(dim=0)
    word_offsets = torch.tensor(word_offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    word = torch.cat(word)
    return text, text_offsets, word, word_offsets, label


class WordDataset(Dataset):
    def __init__(self, json_path='/home/palm/PycharmProjects/wordset-dictionary/data/'):
        self.json_path = json_path
        counter = Counter()
        self.data = []
        for file in os.listdir(json_path):
            data = json.load(open(os.path.join(json_path, file)))
            for key in data:
                word = data[key]
                counter.update([key])
                if 'meanings' in word:
                    self.data.append(word)
                    for meaning in word['meanings']:
                        counter.update(meaning['def'].split(' '))
        self.vocab = Vocab(counter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        diff = torch.rand(1)[0] > 0.5
        if not diff:
            tokens = data['meanings'][0]['def'].split(' ')
            token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                   for token in tokens]))
        else:
            while True:
                idx = torch.randint(0, len(self), (1, ))
                if idx != index:
                    break
            diff_data = self.data[idx]
            tokens = diff_data['meanings'][0]['def'].split(' ')
            token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                   for token in tokens]))

        tokens = torch.tensor(token_ids)
        word = torch.tensor([self.vocab[data['word']]])
        return tokens, word, diff


if __name__ == '__main__':
    t = WordDataset()
    t[0]
