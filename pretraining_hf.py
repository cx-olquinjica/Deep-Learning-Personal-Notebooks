#! /Users/admin/miniconda3/envs/huggingface/bin/python

import torch
import datasets
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset


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

# downloading dataset from the hub 

raw_dataset = load_dataset("glue", "mrpc")
print(raw_dataset)

# raw_dataset returns a dict which contains 
# a train, test, and validation tsst
# load_dataset downloads and caches the dataset at ~/.cache/huggingface/datasets. 

raw_train_dataset = raw_dataset["train"]
print(raw_train_dataset[0])


# preprocss a dataset

tokenized_dataset = tokenizer(
        raw_datasets["train"]["sentence1"],
        raw_datasets["train"]["sentence2"],
        padding=True,
        truncation=True,
        return_tensors="pt"
        ]

# although this is a very intuitive process, and works very well once you know the data structure of your dataset
# it is not very scalable, and that is not so good, if
        
