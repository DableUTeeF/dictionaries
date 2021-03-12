from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset
from models import TextSentiment, ContrastiveLoss, AutoEncoder, TransformerModel
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


if __name__ == '__main__':
    device = 'cpu'
    dataset = WordDataset()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    model = TransformerModel(dataset.vocab_len, dataset.out_vocab_len, 1024)
    state = torch.load('/media/palm/BiggerData/dictionaries/cp2/31_0.000277.pth', map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    vocabs = len(dataset.out_vocab)
    for idx in valid_sampler:
        word, pos_tokens = dataset[idx]

        memory = model.transformer.encoder(model.pos_encoder(model.encoder(pos_tokens.unsqueeze(1).to(device))))
        out_indexes = [dataset.out_vocab.stoi['<sos>']]
        for i in range(2):
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
            output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
            out_token = output.argmax(2)[-1].item()
            out_indexes.append(out_token)
            if out_token == dataset.out_vocab.stoi['<eos>']:
                break

        y_text = model(pos_tokens.unsqueeze(1).to(device), word.unsqueeze(1).to(device))
        print(' '.join([dataset.vocab.itos[i] for i in pos_tokens]))
        print(dataset.out_vocab.itos[out_indexes[1]], dataset.out_vocab.itos[word[1]])
        print(dataset.out_vocab.itos[torch.argmax(y_text, 2)[0]], dataset.out_vocab.itos[word[1]])
        # print(torch.max(y_text, 2)[0])
        # print(torch.argmax(y_text, 2))
