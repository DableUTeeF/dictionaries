from datagen import QuoraDataset, SynonymsDataset, WordTriplet
from models import TextSentiment, ContrastiveLoss, AutoEncoder
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


if __name__ == '__main__':
    device = 'cuda'
    dataset = WordTriplet()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    model = AutoEncoder(len(dataset.vocab), 1024)
    model.to(device)
    vocabs = len(dataset.vocab)
    for (word, pos_tokens, neg_tokens) in enumerate(valid_sampler):
        y_text = model(pos_tokens.to(device))
        print(torch.argmax(y_text, 1))
