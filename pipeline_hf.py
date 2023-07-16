#! /Users/admin/miniconda3/envs/huggingface/bin/python

import transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer 
from transformers import AutoModel 
from transformers import AutoModelForSequenceClassification
from torch.nn import functional as F



classifier = pipeline("sentiment-analysis")
print(classifier("Today is a very beautiful day"))

# data preprocessing ::: model ::: post processing 

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# the tokenizer will return a dictionary 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# once we have the tokenizer we can directly send out data or sentences to it

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# an important thing to have into consideration here, the return_tensor
# it can return a pytorch or a tensorflow tensor (tf)


# Download pretrained model 

model = AutoModel.from_pretrained(checkpoint)
print(model)

# this contains only the Tansfomrer architecture, given some input and it will 
# return hidden_states also known as features 

# these hidden_states returned by the model are usally input to another part of the model
# known as the head. the hidden state mostly returns : 
# the batchF_size, sequence_length, hidden_size : the vector dimension of each model input

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)


# Model Heads: Making sense out of numbers
# take the high-dim vectors of hidden states and project them onto a different dimension

# the output of the Transformer model is sent directly to the head to be processed 

"""HuggingFace has already models or transformers with built-in heads for example: 
    AutoModelForCasualLM
    AutoModelForMaskedLM
    AutoModelForMultipleChoice
    AutoModelForQuestionAnswering
    AutoModelForSequenceClassification
    AutoModelForTokenClassification
    and others...
"""

classification = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = classification(**inputs)
print(outputs.logits.shape)


# Postprocessing the Output: 
print(outputs.logits)

# converting th lgits to probabilities throught softmax 

predictions = F.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)
