from datagen import QuoraDataset, SynonymsDataset, WordTriplet, WordDataset, BertDataset
from models import *
import torch
import numpy as np
from transformers import BertModel, BertTokenizer


if __name__ == '__main__':
    device = 'cpu'
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.requires_grad_(False)
    bert.to(device)
    bert.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertAutoEncoder(tokenizer.vocab_size)
    pth = '/media/palm/BiggerData/dictionaries/cp10/ 4100.pth'
    print(pth)
    state = torch.load(pth)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    inputs = torch.load('/media/palm/BiggerData/dictionaries/cp10/input.pth')

    word, pos_tokens = inputs['word'], inputs['pos_token']

    memories = bert(**pos_tokens.to(device)).last_hidden_state.transpose(0, 1)
    for index in range(32):
        memory = memories[:, index]
        out_indexes = [101]
        for i in range(6):
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(0).to(device)
            embeded_word = bert.embeddings(trg_tensor, token_type_ids=torch.zeros_like(trg_tensor)).transpose(0, 1)
            output = model.fc(model.transformer_decoder(embeded_word, memory))
            out_token = output.argmax(2)[-1].item()
            # print(torch.max(output, 2)[0])
            out_indexes.append(out_token)
            if out_token == 102:
                break
        print(tokenizer.decode(pos_tokens.data['input_ids'][index]))
        print(tokenizer.decode(out_indexes), tokenizer.decode(word.data['input_ids'][index]))
