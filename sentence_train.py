import sys
sys.path.extend(['/home/palm/PycharmProjects/sentence-transformers'])
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from datagen import *
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


if __name__ == '__main__':
    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

    # Define your train examples. You need more than just two examples...
    dataset = ThaiSentenceDataset()

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(dataset.train(True), batch_size=16, shuffle=True)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dataset.train(False))

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=10,
              warmup_steps=100,
              evaluator=evaluator,
              output_path='/media/palm/BiggerData/dictionaries/cp11')
