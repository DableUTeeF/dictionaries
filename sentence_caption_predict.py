from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from datagen import *
import torch
from models import *
import copy


if __name__ == '__main__':
    device = 'cuda'
    eng_sm = SentenceTransformer('cp10-work')
    dataset = SentenceTokenized(eng_sm.tokenizer, 'first', language='eng', true_only=True)
    val_set = dataset.train(False)
    embeddings = copy.deepcopy(eng_sm._first_module().auto_model.embeddings).to(device)

    model = AEPretrainedEmbedding(dataset.vocab_size, embeddings).to(device)
    model.load_state_dict(torch.load('cp14/099_6.3634e-06.pth'))
    with torch.no_grad():
        for data in val_set:
            # meaning, word = data.texts
            word, meaning, _ = dataset.collate_fn([data])
            memory = eng_sm.encode(word, convert_to_tensor=True).unsqueeze(0)
            out_indexes = [dataset.cls]
            for i in range(6):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.embeddings(trg_tensor)
                output = model.transformer_decoder(output, memory)
                output = model.fc(output)
                out_token = output.argmax(2)[-1].item()
                # print(torch.max(output, 2)[0])
                out_indexes.append(out_token)
                if out_token == dataset.sep:
                    break
            print(meaning)
            print(dataset.tokenizer.decode(out_indexes), word)

            # print(meaning, '-', word)
            # print('predict:', score.item(), 'label:', label)
