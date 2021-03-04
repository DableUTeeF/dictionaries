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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                               sampler=train_sampler,
                                               num_workers=1,
                                               collate_fn=dataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                    sampler=valid_sampler,
                                                    num_workers=1,
                                                    collate_fn=dataset.collate_fn)
    model = AutoEncoder(len(dataset.vocab), 1024)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = torch.nn.BCEWithLogitsLoss()
    vocabs = len(dataset.vocab)
    for epoch in range(300):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (word, pos_tokens, neg_tokens) in enumerate(train_loader):
            y_text = model(pos_tokens.to(device))
            target = torch.nn.functional.one_hot(word[0], num_classes=vocabs).float()
            loss = criterion(y_text, target.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progbar.update(idx + 1, [('loss', loss.detach().item()),
                                     ('current_loss', loss.detach().item())])
        schedule.step(progbar._values['loss'][0]/progbar._values['loss'][1])
        model.eval()
        progbar = tf.keras.utils.Progbar(len(validation_loader),
                                         stateful_metrics=['current_loss'])
        min_loss = 100
        with torch.no_grad():
            for idx, (word, pos_tokens, neg_tokens) in enumerate(validation_loader):
                y_text = model(pos_tokens.to(device))
                target = torch.nn.functional.one_hot(word[0], num_classes=vocabs).float()
                loss = criterion(y_text, target.to(device))
                progbar.update(idx + 1, [('val_loss', loss.detach().item()),
                                         ('current_loss', loss.detach().item())])
        # if progbar._values['val_loss'][0]/progbar._values['val_loss'][1] < min_loss:
        #     torch.save(model.state_dict(), f"cp/{progbar._values['val_loss'][0]/progbar._values['val_loss'][1]:.6f}.pth")
