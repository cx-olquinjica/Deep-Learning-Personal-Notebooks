#! /Users/admin/miniconda3/envs/huggingface/bin/python

import torch
import datasets
import evaluate
import numpy as np
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


training_args = TrainingArguments("test-trainer", use_mps_device=True)

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# this is the implementation without the evaluation criteria
"""trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()"""

################################################################################################################

# Evaluation

# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)

# preds = np.argmax(predictions.predictions, axis=-1)

# metric = evaluate.load("glue", "mrpc")

# print(metric.compute(predictions=preds, references=predictions.label_ids))


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", use_mps_device=True)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# to any curious soul out there, these are the results: 

{'eval_loss': 0.733721911907196, 'eval_accuracy': 0.8357843137254902, 'eval_f1': 0.8854700854700854, 'eval_runtime': 3.4403, 'eval_samples_per_second': 118.595, 'eval_steps_per_second': 14.824, 'epoch': 3.0}
{'train_runtime': 254.6969, 'train_samples_per_second': 43.204, 'train_steps_per_second': 5.406, 'train_loss': 0.37109726273502747, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1377/1377 [04:14<00:00,  5.41it/s]
