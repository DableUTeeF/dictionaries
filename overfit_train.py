from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset, BertDataset
from models import *
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from transformers import BertModel
import os


if __name__ == '__main__':
    device = 'cuda'
    dataset = BertDataset()
    vocabs = dataset.vocab_size
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
                                               num_workers=int(device=='cuda'),
                                               collate_fn=dataset.collate_fn,
                                               )
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                    sampler=valid_sampler,
                                                    num_workers=1,
                                                    collate_fn=dataset.collate_fn,
                                                    )
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.requires_grad_(False)
    bert.to(device)
    model = BertAutoEncoder(dataset.vocab_size)

    # state = torch.load('/media/palm/BiggerData/dictionaries/cp7/03_1.3419e-05.pth')
    # model.load_state_dict(state)

    # model = TransformerModel(dataset.vocab_len, dataset.vocab_len, 1024)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(logits=True)
    for idx, (word, pos_tokens,) in enumerate(train_loader):
        break
    torch.save({'word': word, 'pos_token': pos_tokens}, '/media/palm/BiggerData/dictionaries/cp1/input.pth')
    src, trg = pos_tokens.to(device), word.to(device)
    progbar = tf.keras.utils.Progbar(40000,
                                     stateful_metrics=['current_loss'])
    for iteration in range(40000):
        model.train()
        memory = bert(**src).last_hidden_state.transpose(0, 1)
        embeded_word = bert.embeddings(trg.data['input_ids'][:, :-1], token_type_ids=trg.data['token_type_ids'][:, :-1]).transpose(0, 1)
        output = model(memory, embeded_word)
        target = torch.nn.functional.one_hot(trg.data['input_ids'][:, 1:], num_classes=vocabs).float()
        # weight = (torch.FloatTensor(*target.size()).uniform_() < 20/vocabs).float() + 1/vocabs
        # weight = torch.zeros_like(target) + 1/vocabs
        # weight[target == 1] = 1
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(output.transpose(0, 1).transpose(1, 2),
        #                                                             target.transpose(1, 2).to(device),
        #                                                             weight.transpose(1, 2).to(device))
        loss = criterion(output.transpose(0, 1), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progbar.update(iteration + 1, [('loss', loss.detach().item()),
                                 ('current_loss', loss.detach().item())])
        if iteration % 2000 == 100:
            torch.save(model.state_dict(), f"/media/palm/BiggerData/dictionaries/cp1/{iteration:05d}.pth")
