from torch.utils.data import DataLoader
from data_utils import NewsDataset
import torch
import pandas as pd
from gcloud_operations import upload_file, upload_folder, download_file
import argparse

def split_data(bucket_name, data_path):

    download_file(bucket_name, data_path)
    
    train_test_split = [0.8, 0.2]
    val_test_split = [0.5, 0.5]

    random_state = 42 

    data = pd.read_json(data_path, lines=True)
    data_train = data.sample(frac=train_test_split[0], random_state=random_state)

    # Select the remaining rows for the second DataFrame
    data_other = data.drop(data_train.index)

    data_train.to_json('train_data.json', orient='records', lines=True)

    data_val = data_other.sample(frac=val_test_split[0], random_state=random_state)

    # Select the remaining rows for the second DataFrame
    data_test = data_other.drop(data_val.index)

    data_val.to_json('val_data.json', orient='records', lines=True)

    data_test.to_json('test_data.json', orient='records', lines=True)

    upload_file('train_data.json', bucket_name)
    upload_file('val_data.json', bucket_name)
    upload_file('test_data.json', bucket_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str)           # positional argument
    parser.add_argument('--data_path', type=str)           # positional argument
    args = parser.parse_args()

    split_data(args.bucket_name, args.data_path)


