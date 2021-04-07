from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from datagen import *
import torch
if __name__ == '__main__':
    eng_sm = SentenceTransformer('/media/palm/BiggerData/dictionaries/cp11-work')
    dataset = SentenceTokenized(eng_sm.tokenizer, 'eng', true_only=True)
    val_set = dataset.train(False)

    with torch.no_grad():
        for data in val_set:
            meaning, word = data.texts
            label = data.label
            sentence_embeddings = eng_sm.encode(data.texts, convert_to_tensor=True)
            score = torch.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], 0)
            print(meaning, '-', word)
            print('predict:', score.item(), 'label:', label)
