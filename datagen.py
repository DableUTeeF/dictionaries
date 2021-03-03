from torch.utils.data import Dataset
from torchtext.vocab import Vocab, Counter
import torch
from nltk.corpus import wordnet as wn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd


def generate_batch(batch):
    label = torch.tensor([entry[2].int() for entry in batch])
    text = [entry[0] for entry in batch]
    word = [entry[1] for entry in batch]
    text_offsets = [0] + [len(entry) for entry in text]
    word_offsets = [0] + [len(entry) for entry in word]
    text = pad_sequence(text)
    word = pad_sequence(word)
    return text, word, label


def generate_triplet_batch(batch):
    word = [entry[0] for entry in batch]
    pos_text = [entry[1] for entry in batch]
    neg_text = [entry[1] for entry in batch]
    word = pad_sequence(word)
    pos_text = pad_sequence(pos_text)
    neg_text = pad_sequence(neg_text)
    return word, pos_text, neg_text


class QuoraDataset(Dataset):
    def __init__(self):
        self.csv = pd.read_csv('/home/palm/PycharmProjects/quora_question_pair/train.csv')
        self.csv.fillna('', inplace=True)
        counter = Counter()
        for _, row in self.csv.iterrows():
            q1 = row['question1']
            q2 = row['question2']
            counter.update(q1.split(' '))
            counter.update(q2.split(' '))
        self.vocab = Vocab(counter)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        q1 = row['question1']
        q2 = row['question2']
        label = row['is_duplicate']
        token_ids_1 = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                 for token in q1.split(' ')]))
        token_ids_2 = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                 for token in q2.split(' ')]))
        token1 = torch.tensor(token_ids_1)
        token2 = torch.tensor(token_ids_2)
        return token1, token2, torch.ones(1) - label


class WordDataset(Dataset):
    def __init__(self):
        self.words = list(set(i for i in wn.words()))
        counter = Counter()
        for word in self.words:
            counter.update([word])
            word = wn.synsets(word)
            for meaning in word:
                counter.update(meaning.definition().split(' '))
        self.vocab = Vocab(counter)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word = self.words[index]
        data = wn.synsets(word)
        diff = torch.rand(1)[0] > 0.5
        if not diff:
            tokens = data[0].definition()
            token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                   for token in tokens.split(' ')]))
        else:
            while True:
                idx = torch.randint(0, len(self), (1,))
                if idx != index:
                    break
            diff_data = wn.synsets(self.words[idx])
            tokens = diff_data[0].definition()
            token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                   for token in tokens.split(' ')]))

        tokens = torch.tensor(token_ids)
        word = torch.tensor([self.vocab[word]])
        return tokens, word, diff


class WordTriplet(WordDataset):
    def __getitem__(self, index):
        word = self.words[index]
        data = wn.synsets(word)
        pos_tokens = data[0].definition()
        pos_token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                   for token in pos_tokens.split(' ')]))
        while True:
            idx = torch.randint(0, len(self), (1,))
            if idx != index:
                break
        diff_data = wn.synsets(self.words[idx])
        neg_tokens = diff_data[0].definition()
        neg_token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token]
                                                                   for token in neg_tokens.split(' ')]))

        neg_tokens = torch.tensor(neg_token_ids)
        pos_tokens = torch.tensor(pos_token_ids)
        word = torch.tensor([self.vocab[word]])
        return word, pos_tokens, neg_tokens


if __name__ == '__main__':
    dataset = WordTriplet()
    dataset[0]
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               sampler=train_sampler,
                                               num_workers=1,
                                               collate_fn=generate_batch)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                    sampler=valid_sampler,
                                                    num_workers=1,
                                                    collate_fn=generate_batch)
    for idx, _ in enumerate(train_loader):
        pass
    print(idx)
    for idx, _ in enumerate(validation_loader):
        pass
    print(idx)
