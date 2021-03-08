import torch
from torch import nn
from torch.nn import functional as F
import math

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


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.hidden = nn.LSTM(embed_dim, embed_dim, 2)
        self.fc = nn.Linear(embed_dim, 2)
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
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, transformer_dim=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, 2, transformer_dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 4)
        self.pos_encoder = PositionalEncoding(embed_dim, 0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, text):
        src = self.embedding(text)
        # print(src.size())
        # src = self.pos_encoder(src)
        # print(src.size())
        text_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        output = self.transformer_encoder(src, text_mask)
        output = output[-1, :, :]
        # output = F.relu(self.hidden(output))  # todo: should try use all these instead of just the last
        y = F.linear(output, self.embedding.weight.data)
        # y = self.decoder(output)
        return y


class AEv2(nn.Module):
    def __init__(self, vocab_size, embed_dim, transformer_dim=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, 2, transformer_dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 4)
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, 2, transformer_dim, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 4)
        self.pos_encoder = PositionalEncoding(embed_dim, 0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, text):
        src = self.embedding(text)
        src = self.pos_encoder(src)
        text_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        output = self.transformer_encoder(src, text_mask)
        output = self.transformer_decoder(output, text_mask)
        output = output[-1, :, :]
        y = F.linear(output, self.embedding.weight.data)
        # y = self.decoder(output)
        return y


class LSTM_AE(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.hidden_units = embed_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.hidden = nn.LSTM(embed_dim, embed_dim, 2)
        self.decode = nn.Linear(embed_dim, embed_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decode.weight.data.uniform_(-initrange, initrange)
        for value in self.hidden.state_dict():
            param = self.hidden.state_dict()[value]
            if 'weight_ih' in value:
                torch.nn.init.orthogonal_(self.hidden.state_dict()[value])
            elif 'weight_hh' in value:
                weight_hh_data_ii = torch.eye(self.hidden_units, self.hidden_units)  # H_Wii
                weight_hh_data_if = torch.eye(self.hidden_units, self.hidden_units)  # H_Wif
                weight_hh_data_ic = torch.eye(self.hidden_units, self.hidden_units)  # H_Wic
                weight_hh_data_io = torch.eye(self.hidden_units, self.hidden_units)  # H_Wio
                weight_hh_data = torch.stack([weight_hh_data_ii, weight_hh_data_if, weight_hh_data_ic, weight_hh_data_io], dim=0)
                weight_hh_data = weight_hh_data.view(self.hidden_units * 4, self.hidden_units)
                self.hidden.state_dict()[value].data.copy_(weight_hh_data)
            elif 'bias' in value:
                torch.nn.init.constant_(self.hidden.state_dict()[value], val=0)
                self.hidden.state_dict()[value].data[self.hidden_units:self.hidden_units * 2].fill_(1)

    def forward(self, text):
        embedded = self.embedding(text)
        x, (hn, cn) = self.hidden(embedded)  # (seq_len, batch, hidden_size)
        x = x[-1, :, :]
        x = F.hardswish(self.decode(x))
        y = F.linear(x, self.embedding.weight.data)
        return y


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

