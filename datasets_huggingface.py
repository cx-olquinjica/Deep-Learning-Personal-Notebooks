#! /Users/admin/miniconda3/envs/huggingface/bin/python

import requests
import transformers, datasets



API_TOKEN = ""
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/valid"


def query():
    response = requests.get(API_URL)
    return response.json()
data = query()
print(data)
