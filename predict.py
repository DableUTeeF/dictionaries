from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset, BertDataset
from models import TextSentiment, ContrastiveLoss, AutoEncoder, TransformerModel, BertAutoEncoder
from torch.utils.data import DataLoader
import torch
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


if __name__ == '__main__':
    device = 'cpu'
    dataset = BertDataset()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    model = BertAutoEncoder(dataset.vocab_size)
    state = torch.load('/media/palm/BiggerData/dictionaries/cp5/09_4.022677e-05.pth')
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for idx in val_indices:
        word, pos_tokens = dataset.collate_fn([dataset[idx]])

        memory = model.bert(**pos_tokens.to(device)).last_hidden_state.transpose(0, 1)
        out_indexes = [101]
        for i in range(6):
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
            output = model.pos_decoder(model.decoder(trg_tensor))
            output = model.fc(model.transformer_decoder(output, memory))
            out_token = output.argmax(2)[-1].item()
            # print(torch.max(output, 2)[0])
            out_indexes.append(out_token)
            if out_token == 102:
                break
        print(dataset.tokenizer.decode(pos_tokens.data['input_ids'][0]))
        print(dataset.tokenizer.decode(out_indexes), dataset.tokenizer.decode(word.data['input_ids'][0]))

        # y_text = model(pos_tokens.unsqueeze(1).to(device), word.unsqueeze(1).to(device))
        # # print([dataset.vocab.itos[i] for i in pos_tokens])
        # print([dataset.vocab.itos[i] for i in torch.argmax(y_text, 2)],
        #       dataset.vocab.itos[word[1]])
        # print()
        # print(torch.max(y_text, 2)[0])
        # print(torch.argmax(y_text, 2))
