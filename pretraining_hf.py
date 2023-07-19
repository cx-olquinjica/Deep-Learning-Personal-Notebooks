#! /Users/admin/miniconda3/envs/huggingface/bin/python

import torch
import datasets
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset


# data preparation
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model preparation

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

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


# load a a dataset from the hub

def tokenize_function(example):
    """This function takes a dictionary 
    (like the items of our dataset) and 
    returns a new dictionary with the keys 
    input_ids, attention_mask, and token_type_ids."""

    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# apply the tokenize function on all our dataset at once

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)


# Dynamic padding

# while defining the tokenize_function didnt use padding=True
# because padding an entire dataset is not computationally 
# efficient. Now I will define a method to pad by batch instead

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(data_collator)

# getting a few samples

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

# refactored

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})


#############################################################################################################


# Training


training_args = TrainingArguments("test-trainer")

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()
