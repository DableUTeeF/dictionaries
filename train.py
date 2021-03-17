from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset, BertDataset
from models import TextSentiment, ContrastiveLoss, AutoEncoder, AEv2, TransformerModel, BertAutoEncoder
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import f1_score


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
                                               num_workers=1,
                                               collate_fn=dataset.collate_fn,
                                               )
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                    sampler=valid_sampler,
                                                    num_workers=1,
                                                    collate_fn=dataset.collate_fn,
                                                    )
    model = BertAutoEncoder(dataset.vocab_size)
    # model = TransformerModel(dataset.vocab_len, dataset.vocab_len, 1024)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(20):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (word, pos_tokens, ) in enumerate(train_loader):
            src, trg = pos_tokens.to(device), word.to(device)
            output = model(src, trg)
            target = torch.nn.functional.one_hot(trg.data['input_ids'][:, 1:], num_classes=vocabs).float()
            loss = criterion(output.transpose(0, 1).transpose(1, 2), target.transpose(1, 2))
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
            for idx, (word, pos_tokens, ) in enumerate(validation_loader):
                src, trg = pos_tokens.to(device), word.to(device)
                output = model(src, trg)
                target = torch.nn.functional.one_hot(trg.data['input_ids'][:, 1:], num_classes=vocabs).float()
                loss = criterion(output.transpose(0, 1).transpose(1, 2), target.transpose(1, 2))
                progbar.update(idx + 1, [('val_loss', loss.detach().item()),
                                         ('current_loss', loss.detach().item())])
        # if epoch % 10 == 1:
            torch.save(model.state_dict(), f"/media/palm/BiggerData/dictionaries/cp5/{epoch:02d}_{progbar._values['val_loss'][0]/progbar._values['val_loss'][1]:.6e}.pth")
