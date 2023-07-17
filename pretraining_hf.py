#! /Users/admin/miniconda3/envs/huggingface/bin/python

import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification


# data preparation
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model preparation

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# inputs
sequences = [
        "I have been waiting for a HuggingFace course whole my life", 
        "I think this movie is amazing"
        ]

batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
batch["labels"] = torch.tensor([1, 1])

# optimization

optimizer = AdamW(model.parameters())

# loss function

loss = model(**batch).loss

loss.backward()
optimizer.step()

