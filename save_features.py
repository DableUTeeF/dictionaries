from sentence_transformers import SentenceTransformer
from datagen import *
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
import os
import pickle as pk


def check_features(features, strings, sm):  # todo: make this more efficient
    output = []
    for string in strings:
        if string not in features:
            features[string] = sm.encode(string, convert_to_tensor=True).unsqueeze(0).cpu()
        output.append(features[string])
    return torch.cat(output).to(device), features

if __name__ == '__main__':
    device = 'cuda'
    root_data = '/media/palm/BiggerData/dictionaries/'
    eng_sm = SentenceTransformer(os.path.join(root_data, 'cp10-work'))
    eng_sm.requires_grad_(False)
    eng_sm.train(False)

    dataset = SentenceTokenized(eng_sm.tokenizer, 'first', language='eng', true_only=True)

    train_loader = DataLoader(dataset.train(True), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    validation_loader = DataLoader(dataset.train(False), batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    progbar = tf.keras.utils.Progbar(len(train_loader))
    features = {}
    for idx, (words, meanings, labels) in enumerate(train_loader):
        fs = eng_sm.encode(words, convert_to_tensor=True).cpu()
        for word, feature in zip(words, fs):
            features[word] = feature
        progbar.update(idx+1)

    progbar = tf.keras.utils.Progbar(len(validation_loader))
    for idx, (words, meanings, labels) in enumerate(validation_loader):
        fs = eng_sm.encode(words, convert_to_tensor=True).cpu()
        for word, feature in zip(words, fs):
            features[word] = feature
        progbar.update(idx + 1)
    torch.save(features, 'features.pth')

