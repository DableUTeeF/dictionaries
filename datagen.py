from torch.utils.data import Dataset
from torchtext.vocab import Vocab, Counter
import torch
from nltk.corpus import wordnet as wn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer

__all__ = ['BertDataset', 'ThaiBertDataset']

def generate_batch(batch):
    text = [entry[0] for entry in batch]
    word = [entry[1] for entry in batch]
    text = pad_sequence(text)
    word = pad_sequence(word)
    return word, text


def generate_bert_batch(batch):
    word = [entry[0]['input_ids'][0] for entry in batch]
    word = pad_sequence(word, batch_first=True)
    text = [entry[1]['input_ids'][0] for entry in batch]
    text = pad_sequence(text, batch_first=True)
    return word, text


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
        self.csv = pd.read_csv('train.csv')
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


class SynonymsDataset(Dataset):
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
        return len(self.vocab.itos) - 2

    def __getitem__(self, index):
        word = self.vocab.itos[index+2]
        ss = wn.synsets(word)
        lemmas = np.random.choice(ss).lemma_names()
        if len(lemmas) > 1 and np.random.rand() > 0.5:
            other = np.random.choice(lemmas)
            label = torch.ones(1)
        else:
            possible_lemmas = []
            for synset in ss:
                possible_lemmas.extend(synset.lemma_names())
            while True:
                idx = torch.randint(0, len(self), (1,))
                if self.vocab.itos[idx+2] not in possible_lemmas:
                    break
            other = self.vocab.itos[idx+2]
            label = torch.zeros(1)
        word = torch.tensor([self.vocab[word]])
        other = torch.tensor([self.vocab[other]])
        return word, other, label


class BertDataset(Dataset):
    def __init__(self):
        # words = list(set(i for i in wn.words()))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        words = [w for w in list(set(i for i in wn.words())) if len(self.tokenizer([w]).data['input_ids'][0]) == 3]
        self.words = []
        for word in words:
            meanings = wn.synsets(word)
            # word = word.replace('_', ' ')
            for meaning in meanings:
                self.words.append((word, meaning.definition()))
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word, text = self.words[index]
        return word, text

    def collate_fn(self, batch):
        text = [entry[1] for entry in batch]
        word = [entry[0] for entry in batch]
        text = self.tokenizer(text, return_tensors='pt', padding=True)
        word = self.tokenizer(word, return_tensors='pt', padding=True)
        # text.data['attention_mask'][text.data['input_ids'] == 102] = 0
        # text.data['input_ids'][text.data['input_ids'] == 102] = 0
        # word.data['attention_mask'][word.data['input_ids'] == 102] = 0
        # word.data['input_ids'][word.data['input_ids'] == 102] = 0
        return word, text


class ThaiBertDataset(Dataset):
    def __init__(self):
        self.patterns = [r'\([^)]*\)', r'\[[^)]*\]', r'&#[a-z\d]*;', r'<\/[a-z\d]{1,6}>', r'<[a-z\d]{1,6}>']
        self.tokenizer = AutoTokenizer.from_pretrained('mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp')
        self.df = pd.read_csv('data/dictdb_th_en.csv', sep=';')
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        word = row.sentry
        text = row.sdef
        return word, text

    def collate_fn(self, batch):
        text = [entry[1] for entry in batch]
        word = [entry[0] for entry in batch]
        text = self.tokenizer(text, return_tensors='pt', padding=True)
        word = self.tokenizer(word, return_tensors='pt', padding=True)
        # text.data['attention_mask'][text.data['input_ids'] == 102] = 0
        # text.data['input_ids'][text.data['input_ids'] == 102] = 0
        # word.data['attention_mask'][word.data['input_ids'] == 102] = 0
        # word.data['input_ids'][word.data['input_ids'] == 102] = 0
        return word, text


class WordDataset(Dataset):
    def __init__(self):
        words = list(set(i for i in wn.words()))
        counter = Counter()
        self.max_len = 0
        for word in words:
            counter.update([word])
            word = wn.synsets(word)
            for meaning in word:
                definition = re.sub(r'\([^)]*\)', '', meaning.definition())
                if len(definition) == 0:
                    continue
                if definition[0] == ' ':
                    definition = definition[1:]
                self.max_len = max(self.max_len, len(definition.split(' ')))
                counter.update(definition.split(' '))
        self.vocab = Vocab(counter, specials=('<unk>', '<pad>', '<sos>', '<eos>'))
        self.vocab_len = len(self.vocab)
        self.meanings = []
        out_counter = Counter()
        for word in words:
            if counter[word] > 3:
                out_counter.update([word])
                self.meanings.extend([(word, i.definition()) for i in wn.synsets(word)])
        self.out_vocab = Vocab(out_counter, specials=('<unk>', '<pad>', '<sos>', '<eos>'))
        self.out_vocab_len = len(self.out_vocab)

    def __len__(self):
        return len(self.meanings)

    def collate_fn(self, batch):
        return generate_batch(batch)

    def __getitem__(self, index):
        word, tokens = self.meanings[index]
        # data = wn.synsets(word)
        token_ids = [self.vocab['<sos>']] + list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token] for token in tokens.split(' ')])) + [self.vocab['<eos>']]

        tokens = torch.tensor(token_ids)
        word = torch.tensor([self.out_vocab['<sos>'], self.out_vocab[word], self.out_vocab['<eos>']])
        return word, tokens


class WordTriplet(WordDataset):
    def __getitem__(self, index):
        word = self.words[index]
        data = wn.synsets(word)
        pos_tokens = np.random.choice(data).definition()
        pos_token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token] for token in pos_tokens.split(' ')]))
        while True:
            idx = torch.randint(0, len(self), (1,))
            if idx != index:
                break
        diff_data = wn.synsets(self.words[idx])
        neg_tokens = diff_data[0].definition()
        neg_token_ids = list(filter(lambda x: x is not Vocab.UNK, [self.vocab[token] for token in neg_tokens.split(' ')]))

        neg_tokens = torch.tensor(neg_token_ids)
        pos_tokens = torch.tensor(pos_token_ids)
        word = torch.tensor([self.vocab[word]])
        neg_out = torch.zeros((self.max_len, ), dtype=torch.long)
        neg_out[:neg_tokens.size(0)] = neg_tokens
        pos_out = torch.zeros((self.max_len, ), dtype=torch.long)
        pos_out[:pos_tokens.size(0)] = pos_tokens
        word_out = torch.zeros((self.max_len, ), dtype=torch.long)
        word_out[0] = self.vocab[word]
        return word, pos_tokens, neg_tokens

    def collate_fn(self, batch):
        return generate_triplet_batch(batch)


if __name__ == '__main__':
    dataset = WordTriplet()
    x = dataset[0]
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
