"""
    cp13: 768 2048 768, cosineloss, betas 0.7 0.999,
    cp14: 768 2048 768, autoencoder, eng->tha
"""
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from models import *
from datagen import *
import torch
from torch.utils.data import DataLoader
import tensorflow as tf

def check_features(features, strings, sm):  # todo: make this more efficient
    output = []
    for string in strings:
        if string not in features:
            features[string] = sm.encode(string, convert_to_tensor=True).unsqueeze(0).cpu()
        output.append(features[string])
    return torch.cat(output).to(device), features

if __name__ == '__main__':
    device = 'cuda'
    dataset = SentenceDataset('all', true_only=True)

    tha_sm = SentenceTransformer('/media/palm/BiggerData/dictionaries/cp11-work')
    tha_sm.requires_grad_(False)
    tha_sm.train(False)
    eng_sm = SentenceTransformer('/media/palm/BiggerData/dictionaries/cp10-work')
    eng_sm.requires_grad_(False)
    eng_sm.train(False)

    model = SentenceMatch(768, 2048, 768)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.7, 0.999))
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-8, verbose=True)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(dataset.train(True), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    validation_loader = DataLoader(dataset.train(False), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    features = {}
    for epoch in range(100):
        print('Epoch:', epoch + 1)
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (eng_words, tha_words, labels) in enumerate(train_loader):
            eng_features, features = check_features(features, eng_words, eng_sm)
            eng_features = model(eng_features.to(device))
            tha_features, features = check_features(features, tha_words, tha_sm)
            # tha_features = model(tha_features.to(device))
            loss = criterion(eng_features, tha_features)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progbar.update(idx + 1, [('loss', loss.detach().item()),
                                     ('current_loss', loss.detach().item())])
        model.eval()
        schedule.step(progbar._values['loss'][0]/progbar._values['loss'][1])
        progbar = tf.keras.utils.Progbar(len(validation_loader),
                                         stateful_metrics=['current_loss'])
        with torch.no_grad():
            for idx, (eng_words, tha_words, labels) in enumerate(validation_loader):
                eng_features, features = check_features(features, eng_words, eng_sm)
                eng_features = model(eng_features.to(device))
                tha_features, features = check_features(features, tha_words, tha_sm)
                # tha_features = model(tha_features.to(device))
                loss = criterion(eng_features, tha_features)
                progbar.update(idx + 1, [('val_loss', loss.detach().item()),
                                         ('current_loss', loss.detach().item())])
        the_loss = progbar._values['val_loss'][0] / progbar._values['val_loss'][1]
        if abs(the_loss) > 1e-3:
            the_loss = f'{the_loss:.4f}'
        else:
            the_loss = f'{the_loss:.4e}'
        torch.save(model.state_dict(), f"/media/palm/BiggerData/dictionaries/cp14/{epoch:03d}_{the_loss}.pth")
