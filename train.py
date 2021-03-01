from datagen import WordDataset, generate_batch
from models import TextSentiment, ContrastiveLoss
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':
    device = 'cuda'
    datagen = WordDataset()
    data = DataLoader(datagen, batch_size=32, shuffle=True,
                      collate_fn=generate_batch)

    model = TextSentiment(len(datagen.vocab), 1024)
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = ContrastiveLoss()
    for epoch in range(30):
        print(epoch)
        for text, text_offsets, word, word_offsets, label in data:
            y_text = model(text.to(device), text_offsets.to(device))
            y_word = model(word.to(device), word_offsets.to(device))
            loss = criterion(y_text, y_word, label.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss)

