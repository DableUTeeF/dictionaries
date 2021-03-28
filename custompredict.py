"""
Working weights
cp9 - BertAutoEncoderOld ~ 19
"""
from datagen import *
from models import *
import torch
import numpy as np
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence


if __name__ == '__main__':
    device = 'cpu'
    bert = BertModel.from_pretrained('monsoon-nlp/bert-base-thai')
    bert.requires_grad_(False)
    bert.to(device)
    bert.eval()
    dataset = RoyinDataset()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(88)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    model = BertAutoEncoderOld(dataset.vocab_size)
    pth = '/media/palm/BiggerData/dictionaries/cp14/039_1.3431e-06.pth'
    print(pth)
    state = torch.load(pth, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    text = ['น. แสงสว่างที่พวยพุ่งออกจากจุดกลาง, แสงสว่าง; เส้นที่ลากจากจุดศูนย์กลางของวงกลมไปถึงเส้นรอบวง. (ส.; ป. รํสิ).']
    print(text[0])
    text = pad_sequence([torch.tensor(dataset.tokenizer(t)) for t in text], True)

    memory = bert(text.to(device)).last_hidden_state.transpose(0, 1)

    out_indexes = [dataset.cls]
    for i in range(6):
        trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
        output = model.pos_decoder(model.decoder(trg_tensor))
        output = model.fc(model.transformer_decoder(output, memory))
        out_token = output.argmax(2)[-1].item()
        # print(torch.max(output, 2)[0])
        out_indexes.append(out_token)
        if out_token == dataset.sep:
            break
    print(dataset.target[out_indexes[1]+2], 'ไม้นิ้ว')

