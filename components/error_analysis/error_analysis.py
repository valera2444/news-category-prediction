

import argparse


import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from gcloud_operations import upload_file, upload_folder, download_file, download_folder

from pathlib import Path

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

import os

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

def create_confusion_matrix(categorical_preds, labels, out_path):


    cm_original = confusion_matrix(labels, categorical_preds)


    cm_norm = confusion_matrix(labels, categorical_preds, normalize='true')

    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(cm_norm,
                ax=ax,
                annot=cm_original,
                fmt='d',
                annot_kws={"size": 7},
                xticklabels=np.unique(np.array(labels)),
                yticklabels=np.unique(np.array(labels)))

    

    plt.savefig(out_path+'confusion_matrix.png')

def crete_metrics(categorical_preds, labels, out_path):



    f1_scores = f1_score(labels, categorical_preds, average=None)
    precision_scores = precision_score(labels, categorical_preds, average=None)
    recall_scores = recall_score(labels, categorical_preds, average=None)


    df = pd.DataFrame({'f1':f1_scores,'precision':precision_scores, 'recall':recall_scores})

    df['category'] = [index_to_category[idx] for idx in range(len(df))]
    df.sort_values(by='f1', ascending=False)


    df.to_csv(out_path+'category_f1.csv')

def main(bucket_name, data_path, out_path):

    Path(out_path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(data_path):
        download_file(bucket_name, data_path)

    f = pd.read_csv(data_path)
    preds, labels = f.labels, f.categorical_preds
    create_confusion_matrix(preds, labels, out_path)
    crete_metrics(preds, labels, out_path)

    upload_file(out_path+'confusion_matrix.png', bucket_name)
    upload_file(out_path + 'category_f1.csv', bucket_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str)           # positional argument
    parser.add_argument('--data_path', type=str)           # positional argument
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()

    main(args.bucket_name, args.data_path, args.out_path)