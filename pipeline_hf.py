#! /Users/admin/miniconda3/envs/huggingface/bin/python

import transformers
from transformers import pipeline
from transformers import AutoTokenizer 


classifier = pipeline("sentiment-analysis")
print(classifier("Today is a very beautiful day"))

# data preprocessing ::: model ::: post processing 

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# once we have the tokenizer we can directly send out data or sentences to it

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# an important thing to have into consideration here, the return_tensor


