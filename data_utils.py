import torch
import pandas as pd
from torch.utils.data import Dataset
import re 
class NewsDataset(Dataset):

    def __preprocess(self, data):
        data = data.fillna('absent')
        
        data.headline[data['headline'].apply(len) == 0] = 'absent'
        data.link[data['link'].apply(len) == 0] = 'absent'
        data.short_description[data['short_description'].apply(len) == 0] = 'absent'
        data.authors[data['authors'].apply(len) == 0] = 'absent'
        return data
        
    def __init__(self, path):
        """_summary_

        Args:
            path (str): path to .json file
        """
        j=pd.read_json(path, lines=True)
        j=self.__preprocess(j)
        
        link = j['link'].str.lower().replace(r'\b(www|http|https|com|html)\b',' ', regex=True)

        
        link = link.apply(lambda l: re.sub('[^a-z A-Z 0-9]+', ' ',l ))

        
        self.x = 'headline: ' +  j['headline'] + \
        ' ; short_description: ' + j['short_description'] + \
        ' ; authors: ' + j['authors'] + \
        ' ; link: ' + link

        
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
    
