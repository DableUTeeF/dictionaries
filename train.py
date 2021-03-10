from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset
from models import TextSentiment, ContrastiveLoss, AutoEncoder, AEv2, TransformerModel
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import f1_score


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
    # model = AEv2(dataset.vocab_len, 1024)
    model = TransformerModel(dataset.vocab_len, dataset.vocab_len, 1024)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    vocabs = dataset.vocab_len
    for epoch in range(20):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (word, pos_tokens, ) in enumerate(train_loader):
            target = torch.nn.functional.one_hot(word[0], num_classes=vocabs).float()
            y_text = model(pos_tokens.to(device), word.to(device))
            weight = (torch.FloatTensor(*target.size()).uniform_() < 20/vocabs).float()
            weight[target == 1] = 1
            # loss = criterion(y_text, target.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_text,
                                                                        target.to(device),
                                                                        weight.to(device))
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
                y_text = model(pos_tokens.to(device), word.to(device))
                target = torch.nn.functional.one_hot(word[0], num_classes=vocabs).float()
                loss = criterion(y_text, target.to(device))
                progbar.update(idx + 1, [('val_loss', loss.detach().item()),
                                         ('current_loss', loss.detach().item())])
        # if epoch % 10 == 1:
        #     torch.save(model.state_dict(), f"/media/palm/BiggerData/dictionaries/cp/{epoch:02d}_{progbar._values['val_loss'][0]/progbar._values['val_loss'][1]:.6f}.pth")
