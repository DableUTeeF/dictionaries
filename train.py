"""
todo: 1). Attention mask - cp8(with eos removed)
      2). Remove EOS - cp3
      3). Extreme large batch size - cp4
      4). Weight EOS and PAD to 0 in loss - cp8 - cp9
"""
from datagen import *
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
                                               num_workers=int(device == 'cuda') * 2,
                                               collate_fn=dataset.collate_fn,
                                               )
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                    sampler=valid_sampler,
                                                    num_workers=2,
                                                    collate_fn=dataset.collate_fn,
                                                    )
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.requires_grad_(False)
    bert.to(device)
    model = BertAutoEncoder(dataset.vocab_size)

    state = torch.load('/media/palm/BiggerData/dictionaries/cp6/03_5.9749e-05.pth')
    model.load_state_dict(state)

    # model = TransformerModel(dataset.vocab_len, dataset.vocab_len, 1024)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(logits=True)
    for epoch in range(100):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (word, pos_tokens,) in enumerate(train_loader):
            src, trg = pos_tokens.to(device), word.to(device)
            memory = bert(**src).last_hidden_state.transpose(0, 1)
            # memory_mask = torch.zeros_like(memory)
            # memory_mask[src.data['attention_mask'].transpose(0, 1) == 1] = 1
            embeded_word = bert.embeddings(trg.data['input_ids'][:, :-1], token_type_ids=trg.data['token_type_ids'][:, :-1]).transpose(0, 1)
            # tgt_mask = torch.zeros_like(embeded_word)
            # tgt_mask[trg.data['attention_mask'][:, :-1].transpose(0, 1)==1] = 1
            output = model(memory, embeded_word)
            target = torch.nn.functional.one_hot(trg.data['input_ids'][:, 1:], num_classes=vocabs).float()
            # weight = (torch.FloatTensor(*target.size()).uniform_() < 20/vocabs).float() + 1/vocabs
            # weight = torch.zeros_like(target) + 1/vocabs
            # weight[target == 1] = 1
            weight = torch.ones_like(target)
            weight[trg.data['attention_mask'][:, 1:] == 0] = 0
            weight[trg.data['input_ids'][:, 1:] == 102] = 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.transpose(0, 1).transpose(1, 2),
                                                                        target.transpose(1, 2).to(device),
                                                                        weight.transpose(1, 2).to(device))
            # loss = criterion(output.transpose(0, 1), target)
            loss /= 32
            loss.backward()
            if (idx + 1) % 32 == 0:
                optimizer.step()
                optimizer.zero_grad()
            progbar.update(idx + 1, [('loss', loss.detach().item()),
                                     ('current_loss', loss.detach().item())])
        optimizer.step()
        optimizer.zero_grad()
        # schedule.step(progbar._values['loss'][0]/progbar._values['loss'][1])
        model.eval()
        progbar = tf.keras.utils.Progbar(len(validation_loader),
                                         stateful_metrics=['current_loss'])
        min_loss = 100
        with torch.no_grad():
            for idx, (word, pos_tokens,) in enumerate(validation_loader):
                src, trg = pos_tokens.to(device), word.to(device)
                memory = bert(**src).last_hidden_state.transpose(0, 1)
                embeded_word = bert.embeddings(trg.data['input_ids'][:, :-1], token_type_ids=trg.data['token_type_ids'][:, :-1]).transpose(0, 1)
                output = model(memory, embeded_word)
                target = torch.nn.functional.one_hot(trg.data['input_ids'][:, 1:], num_classes=vocabs).float()
                weight = torch.ones_like(target)
                weight[trg.data['attention_mask'][:, 1:] == 0] = 0
                weight[trg.data['input_ids'][:, 1:] == 102] = 0
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output.transpose(0, 1).transpose(1, 2),
                                                                            target.transpose(1, 2).to(device),
                                                                            weight.transpose(1, 2).to(device))
                # loss = criterion(output.transpose(0, 1).transpose(1, 2), target.transpose(1, 2))
                progbar.update(idx + 1, [('val_loss', loss.detach().item()),
                                         ('current_loss', loss.detach().item())])
            # if epoch % 10 == 1:
            the_loss = progbar._values['val_loss'][0] / progbar._values['val_loss'][1]
            if abs(the_loss) > 1e-3:
                the_loss = f'{the_loss:.4f}'
            else:
                the_loss = f'{the_loss:.4e}'
            torch.save(model.state_dict(), f"/media/palm/BiggerData/dictionaries/cp8/{epoch:02d}_{the_loss}.pth")
        # torch.save(model.state_dict(), f"/media/palm/BiggerData/dictionaries/cp3/last.pth")
