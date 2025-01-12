import torch
import pandas as pd
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, path):
        """_summary_

        Args:
            path (str): path to .json file
        """
        j=pd.read_json(path, lines=True)
        self.x = 'author: ' + j['headline'] + '; headline: ' +  j['headline'] + '; short_description: ' + j['short_description']
        self.y = j['category'].astype('category').cat.codes
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """

        Args:
            idx (list): _description_

        Returns:
            str: single item
        """
        return {'x':self.x.iloc[idx],'y':self.y.iloc[idx]}
    
