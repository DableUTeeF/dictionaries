"""
Working weights
cp9 - BertAutoEncoderOld ~ 19
"""
from datagen import *
from models import *
import torch
import numpy as np
from transformers import AutoModel


if __name__ == '__main__':
    device = 'cpu'
    bert = AutoModel.from_pretrained('roberta-large')
    bert.requires_grad_(False)
    bert.to(device)
    bert.eval()
    dataset = BertDataset(name='roberta-large', bos='<s>', eos='</s>')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    model = BertAutoEncoderOld(dataset.vocab_size, 1024)
    pth = '/media/palm/BiggerData/dictionaries/cp10/042_6.8085e-06.pth'
    print(pth)
    state = torch.load(pth, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    for idx in val_indices:
        word, pos_tokens = dataset.collate_fn([dataset[idx]])

        memory = bert(**pos_tokens.to(device)).last_hidden_state.transpose(0, 1)
        out_indexes = [dataset.cls]
        for i in range(6):
            # trg_tensor = torch.LongTensor(out_indexes).unsqueeze(0).to(device)
            # embeded_word = bert.embeddings(trg_tensor, token_type_ids=torch.zeros_like(trg_tensor)).transpose(0, 1)
            # output = model.fc(model.transformer_decoder(embeded_word, memory))
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
            output = model.pos_decoder(model.decoder(trg_tensor))
            output = model.fc(model.transformer_decoder(output, memory))
            out_token = output.argmax(2)[-1].item()
            # print(torch.max(output, 2)[0])
            out_indexes.append(out_token)
            if out_token == dataset.sep:
                break
        # print(''.join(dataset.tokenizer.decode(pos_tokens[0].tolist())))
        # print(([dataset.target[i] for i in out_indexes]), ([dataset.target[i] for i in word[0]]))
        print(dataset.tokenizer.decode(pos_tokens.data['input_ids'][0]))
        print(dataset.tokenizer.decode(out_indexes), dataset.tokenizer.decode(word.data['input_ids'][0]))

        # y_text = model(pos_tokens.unsqueeze(1).to(device), word.unsqueeze(1).to(device))
        # # print([dataset.vocab.itos[i] for i in pos_tokens])
        # print([dataset.vocab.itos[i] for i in torch.argmax(y_text, 2)],
        #       dataset.vocab.itos[word[1]])
        # print()
        # print(torch.max(y_text, 2)[0])
        # print(torch.argmax(y_text, 2))
