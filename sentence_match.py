from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from models import *
from datagen import *
import torch
from torch.utils.data import DataLoader
import tensorflow as tf


if __name__ == '__main__':
    dataset = SentenceDataset('all')

    tha_sm = SentenceTransformer('cp11-work')
    eng_sm = SentenceTransformer('cp10-work')

    model = SentenceMatch(768, 2048, 768)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-8, verbose=True)
    criterion = CosineLoss()

    train_loader = DataLoader(dataset.train(True), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(dataset.train(False), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    for epoch in range(30):
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (eng_words, tha_words, labels) in enumerate(train_loader):
            eng_features = eng_sm.encode(eng_words, convert_to_tensor=True)
            tha_features = tha_sm.encode(tha_words, convert_to_tensor=True)
            loss = criterion(eng_features, tha_features, labels)
