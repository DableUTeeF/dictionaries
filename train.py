from datagen import QuoraDataset, generate_batch
from models import TextSentiment, ContrastiveLoss
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


if __name__ == '__main__':
    device = 'cuda'
    dataset = QuoraDataset()
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
    model = TextSentiment(len(dataset.vocab), 1024)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    criterion = ContrastiveLoss()
    for epoch in range(30):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (text, word, label) in enumerate(train_loader):
            y_text = model(text.to(device))
            y_word = model(word.to(device))
            loss = criterion(y_text, y_word, label.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progbar.update(idx + 1, [('loss', loss.detach().item())])
        schedule.step(progbar._values['loss'][0]/progbar._values['loss'][1])
        model.eval()
        progbar = tf.keras.utils.Progbar(len(validation_loader))
        min_loss = 100
        with torch.no_grad():
            for idx, (text, word, label) in enumerate(validation_loader):
                y_text = model(text.to(device))
                y_word = model(word.to(device))
                loss = criterion(y_text, y_word, label.to(device))
                progbar.update(idx + 1, [('val_loss', loss.detach().item())])
        # if progbar._values['val_loss'][0]/progbar._values['val_loss'][1] < min_loss:
        #     torch.save(model.state_dict(), f"cp/{progbar._values['val_loss'][0]/progbar._values['val_loss'][1]:.6f}.pth")
