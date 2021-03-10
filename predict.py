from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset
from models import TextSentiment, ContrastiveLoss, AutoEncoder, TransformerModel
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


if __name__ == '__main__':
    device = 'cuda'
    dataset = WordDataset()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    model = TransformerModel(dataset.vocab_len, dataset.vocab_len, 1024)
    state = torch.load('01_0.000197.pth')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    vocabs = len(dataset.vocab)
    for idx in valid_sampler:
        word, pos_tokens = dataset[idx]
        y_text = model(pos_tokens.unsqueeze(1).to(device), word.unsqueeze(1).to(device))
        print([dataset.vocab.itos[i] for i in pos_tokens])
        print([dataset.vocab.itos[i] for i in torch.argmax(y_text, 2)])
        print(torch.max(y_text, 2)[0])
        print(torch.argmax(y_text, 2))
