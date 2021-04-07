from sentence_transformers import SentenceTransformer
from models import *
from datagen import *
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
import os
import copy


def check_features(features, strings, sm):  # todo: make this more efficient
    output = []
    for string in strings:
        if string not in features:
            features[string] = sm.encode(string, convert_to_tensor=True).unsqueeze(0).cpu()
        output.append(features[string])
    return torch.cat(output).to(device), features


if __name__ == '__main__':
    device = 'cuda'
    if os.path.isdir('/media/palm/BiggerData/dictionaries/'):
        root_data = '/media/palm/BiggerData/dictionaries/'
    elif os.path.isdir('/home/palm/PycharmProjects/cp/cp10-work/'):
        root_data = '/home/palm/PycharmProjects/cp'
    elif os.path.isdir('/home/palm/PycharmProjects/nlp/cp10-work'):
        root_data = '/home/palm/PycharmProjects/nlp/'
    else:
        raise ValueError('Well, something\'s wrong here')
    eng_sm = SentenceTransformer(os.path.join(root_data, 'cp10-work'))
    eng_sm.requires_grad_(False)
    eng_sm.train(False)

    embeddings = copy.deepcopy(eng_sm._first_module().auto_model.embeddings).to(device)
    dataset = SentenceTokenized(eng_sm.tokenizer, 'first', language='eng', true_only=True)

    model = AEPretrainedEmbedding(dataset.vocab_size, embeddings)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.7, 0.999))
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=1e-8, verbose=True)
    criterion = FocalLoss()

    train_loader = DataLoader(dataset.train(True), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    validation_loader = DataLoader(dataset.train(False), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    features = {}
    vocabs = dataset.vocab_size
    for epoch in range(100):
        print('Epoch:', epoch + 1)
        progbar = tf.keras.utils.Progbar(len(train_loader),
                                         stateful_metrics=['current_loss'])
        for idx, (words, meanings, labels) in enumerate(train_loader):
            meanings = meanings.to(device)
            words, features = check_features(features, words, eng_sm)
            words_features = model(words.to(device), meanings)
            target = torch.nn.functional.one_hot(meanings.data['input_ids'][:, 1:], num_classes=vocabs).float()
            loss = criterion(words_features.transpose(0, 1).transpose(1, 2), target.transpose(1, 2))
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
            for idx, (words, meanings, labels) in enumerate(validation_loader):
                meanings = meanings.to(device)
                words, features = check_features(features, words, eng_sm)
                words_features = model(words.to(device), meanings)
                target = torch.nn.functional.one_hot(meanings.data['input_ids'][:, 1:], num_classes=vocabs).float()
                loss = criterion(words_features.transpose(0, 1).transpose(1, 2), target.transpose(1, 2))
                progbar.update(idx + 1, [('val_loss', loss.detach().item()),
                                         ('current_loss', loss.detach().item())])
        the_loss = progbar._values['val_loss'][0] / progbar._values['val_loss'][1]
        if abs(the_loss) > 1e-3:
            the_loss = f'{the_loss:.4f}'
        else:
            the_loss = f'{the_loss:.4e}'
        os.makedirs(os.path.join(root_data, 'cp14'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(root_data, 'cp14', f"{epoch:03d}_{the_loss}.pth"))
