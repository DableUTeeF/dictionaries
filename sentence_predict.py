from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from datagen import SentenceDataset
import torch
if __name__ == '__main__':
    model = SentenceTransformer('/media/palm/BiggerData/dictionaries/cp10')
    dataset = SentenceDataset()
    val_set = dataset.train(False)

    sentences = ['The cos_score_transformation function is applied on top of cosine_similarity',
                 'School']
    with torch.no_grad():
        for data in val_set:
            meaning, word = data.texts
            label = data.label
            sentence_embeddings = model.encode(data.texts, convert_to_tensor=True)
            score = torch.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], 0)
            print(meaning, '-', word)
            print('predict:', score.item(), 'label:', label)
