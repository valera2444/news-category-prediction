import torch
from transformers import AutoTokenizer
from data_utils import NewsDataset

from torch.utils.data import DataLoader

import argparse

def load_model(path):
    model = torch.load('weights18_final.pt', weights_only=False)
    model.eval()
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
def create_dataloader(path, batch_size):

        
    ds = NewsDataset(path)
    test_dataloader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    return test_dataloader

def predict(model, tokenizer, test_dataloader, device):
    predictions = []
    with torch.no_grad():
            
        for b_eval in test_dataloader:
            tokenized = tokenizer(b_eval['x'], padding=True, return_tensors="pt")
            tokenized.to(device)
            ids, mask, token_type_ids = tokenized['input_ids'], tokenized['attention_mask'], tokenized['token_type_ids']
            preds = model(ids, mask, token_type_ids)
            predictions.append( torch.argmax(preds, dim=1))

    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)           # positional argument
    parser.add_argument('model_path', type=str)      # option that takes a value
    parser.add_argument('batch_size', type=int) 
    parser.add_argument('device', type=int, choices=['cuda','cpu']) 
    args = parser.parse_args()

    if args.device == 'cuda':
        assert torch.cuda().is_availbale()
    
    model = load_model(args.model_path)
    tokenizer = load_tokenizer()
    dataloader = create_dataloader(args.data_path, args.batch_size)

    preds = predict(model, tokenizer, dataloader, args.device)
    print(preds)
