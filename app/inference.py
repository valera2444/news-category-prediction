import torch
from transformers import AutoTokenizer
from data_utils import NewsDataset

from torch.utils.data import DataLoader

import argparse

from model import Transformer, CustomBertForClassification

import pandas as pd
import numpy as np

from  gcloud_operations import download_file

import os.path


#dict( enumerate(j['category'].astype('category').cat.categories ) ) on NewsDataset(data.json)
index_to_category = {
    0: 'ARTS',
    1: 'ARTS & CULTURE',
    2: 'BLACK VOICES',
    3: 'BUSINESS',
    4: 'COLLEGE',
    5: 'COMEDY',
    6: 'CRIME',
    7: 'CULTURE & ARTS',
    8: 'DIVORCE',
    9: 'EDUCATION',
    10: 'ENTERTAINMENT',
    11: 'ENVIRONMENT',
    12: 'FIFTY',
    13: 'FOOD & DRINK',
    14: 'GOOD NEWS',
    15: 'GREEN',
    16: 'HEALTHY LIVING',
    17: 'HOME & LIVING',
    18: 'IMPACT',
    19: 'LATINO VOICES',
    20: 'MEDIA',
    21: 'MONEY',
    22: 'PARENTING',
    23: 'PARENTS',
    24: 'POLITICS',
    25: 'QUEER VOICES',
    26: 'RELIGION',
    27: 'SCIENCE',
    28: 'SPORTS',
    29: 'STYLE',
    30: 'STYLE & BEAUTY',
    31: 'TASTE',
    32: 'TECH',
    33: 'THE WORLDPOST',
    34: 'TRAVEL',
    35: 'U.S. NEWS',
    36: 'WEDDINGS',
    37: 'WEIRD NEWS',
    38: 'WELLNESS',
    39: 'WOMEN',
    40: 'WORLD NEWS',
    41: 'WORLDPOST'
 }

def load_model(path,tokenizer):
    t = Transformer(vocab_size=tokenizer.vocab_size, model_dim=768,n_heads=12,max_seq_len=512,n_blocks=12)
    model = CustomBertForClassification(t, 1024, 42)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))

    model.eval()
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer

def create_dataloader(headline, link, short_description, authors):

    path='data.json'

    df = pd.DataFrame({'headline':headline,
                       'link':link,
                       'short_description':short_description,
                       'authors':authors}, index=[0])
    
    df.to_json(path, orient='records', lines=True)

    ds = NewsDataset(path)
    test_dataloader = DataLoader(ds, batch_size=1,
                        shuffle=False, num_workers=0)
    return test_dataloader

def predict(model, tokenizer, test_dataloader):
    predictions = []
    with torch.no_grad():
            
        for b_eval in test_dataloader:
            tokenized = tokenizer(b_eval['x'], padding=True, return_tensors="pt")
            tokenized
            ids, mask, token_type_ids = tokenized['input_ids'], tokenized['attention_mask'], tokenized['token_type_ids']
            preds = model(ids, mask, token_type_ids)
            predictions.append( torch.argmax(preds, dim=1))

    predictions = torch.cat(predictions).numpy(force=True)

    categorical_preds  = [index_to_category[idx] for idx in predictions]

    return categorical_preds[0]

def main(headline, link, short_description, authors):

    download_file('news-category-prediction','weights_final_81_acc.pt', 'app/weights_final_81_acc.pt')

    model_path='app/weights_final_81_acc.pt'
    tokenizer = load_tokenizer()
    model = load_model(model_path, tokenizer)
    dataloader = create_dataloader(headline, link, short_description, authors)

    preds = predict(model, tokenizer, dataloader)

    return preds


#import pandas as pd
#s = pd.read_json('data.json', lines=True)
#s[0:50].to_json('data_for_inference.json', orient='records',lines=True)