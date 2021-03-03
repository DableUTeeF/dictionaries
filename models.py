import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.float()
        euclidean_distance = F.pairwise_distance(output1, output2)
        # cosine_similarity = F.cosine_similarity(output1, output2)
        # euclidean_distance = 1/cosine_similarity
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.hidden = nn.LSTM(embed_dim, embed_dim, 2)
        self.fc = nn.Linear(embed_dim, 128)
        # self.fc2 = nn.Linear(1024, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        x, (hn, cn) = self.hidden(embedded)  # (seq_len, batch, hidden_size)
        x = x[-1, :, :]
        # x = self.fc(x.view(x.size(1), x.size(2)))
        # x = self.fc(embedded)
        x = self.fc(x)
        print(x.size())
        # x = self.fc2(x)
        return x


class TripletModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.hidden = nn.LSTM(embed_dim, embed_dim, 2)
        self.fc = nn.Linear(embed_dim, 128)


if __name__ == '__main__':
    from datagen import WordDataset, generate_batch
    from torch.utils.data import DataLoader

    datagen = WordDataset()
    data = DataLoader(datagen, batch_size=6, shuffle=True,
                      collate_fn=generate_batch)

    model = TextSentiment(len(datagen.vocab), 1024)
    with torch.no_grad():
        for text, text_offsets, word, word_offsets, label in data:
            y = model(text, text_offsets)
    print()

