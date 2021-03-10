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
    state = torch.load('/home/palm/PycharmProjects/nlp/10_0.000011.pth')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    vocabs = len(dataset.vocab)
    for idx in valid_sampler:
        word, pos_tokens = dataset[idx]
        y_text = model(pos_tokens.unsqueeze(1).to(device), word.unsqueeze(1).to(device))
        print(torch.argmax(y_text, 1))
        print([dataset.vocab.itos[i] for i in torch.argmax(y_text, 1)])
