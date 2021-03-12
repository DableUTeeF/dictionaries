from transformers import BertTokenizer, BertModel


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = 'a '
output = tokenizer(text, return_tensors="pt")
