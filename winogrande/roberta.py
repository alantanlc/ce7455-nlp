from transformers import RobertaTokenizer, RobertaForTokenClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForTokenClassification.from_pretrained('roberta-base')

s = 'Hello, my dog is cute'
encoded_tokens = tokenizer.encode(s, add_special_tokens=True)
input_ids = torch.tensor(encoded_tokens).unsqueeze(0) # Batch size 1
labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0) # Batch size 1

outputs = model(input_ids, labels=labels)
loss, scores = outputs[:2]