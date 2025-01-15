import argparse

import torch.nn as nn
from model import Transformer, CustomBertForClassification

import torch

from transformers import BertModel

import matplotlib.pyplot as pl
from transformers import AutoTokenizer


from torch.utils.data import DataLoader
from data_utils import NewsDataset

import numpy as np
import matplotlib.pyplot as plt

from gcloud_operations import download_file, download_folder, upload_file, upload_folder

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def prepare_model():

    device = 'cuda:0'if torch.cuda.is_available() else 'cpu'
    device
    t = Transformer(vocab_size=tokenizer.vocab_size, model_dim=768,n_heads=12,max_seq_len=512,n_blocks=12)
    
    model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
   
    load_custom_model_weights(model, t)
    classification_bert = CustomBertForClassification(t, 1024, 42)

    classification_bert.to(device)
    

    return classification_bert

# Function to load weights from base BERT into the custom model
def load_custom_model_weights(base_model, custom_model):
    custom_model_state_dict = custom_model.state_dict()
    base_model_state_dict = base_model.state_dict()

    # Mapping for the embedding layers
    mapping = {
        "emb.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "emb.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "emb.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "emb_ln.gamma": "embeddings.LayerNorm.weight",
        "emb_ln.beta": "embeddings.LayerNorm.bias"
    }

    # Mapping for the encoder layers
    for i in range(12):  # Loop through the 12 transformer layers
        mapping.update({
            f"net.{i}.mha.qkv.weight": [
                f"encoder.layer.{i}.attention.self.query.weight",
                f"encoder.layer.{i}.attention.self.key.weight",
                f"encoder.layer.{i}.attention.self.value.weight"
            ],
            f"net.{i}.mha.qkv.bias": [
                f"encoder.layer.{i}.attention.self.query.bias",
                f"encoder.layer.{i}.attention.self.key.bias",
                f"encoder.layer.{i}.attention.self.value.bias"
            ],
            f"net.{i}.mha.proj.weight": f"encoder.layer.{i}.attention.output.dense.weight",
            f"net.{i}.mha.proj.bias": f"encoder.layer.{i}.attention.output.dense.bias",
            f"net.{i}.ln1.gamma": f"encoder.layer.{i}.attention.output.LayerNorm.weight",
            f"net.{i}.ln1.beta": f"encoder.layer.{i}.attention.output.LayerNorm.bias",
            f"net.{i}.mlp.0.weight": f"encoder.layer.{i}.intermediate.dense.weight",
            f"net.{i}.mlp.0.bias": f"encoder.layer.{i}.intermediate.dense.bias",
            f"net.{i}.mlp.2.weight": f"encoder.layer.{i}.output.dense.weight",
            f"net.{i}.mlp.2.bias": f"encoder.layer.{i}.output.dense.bias",
            f"net.{i}.ln2.gamma": f"encoder.layer.{i}.output.LayerNorm.weight",
            f"net.{i}.ln2.beta": f"encoder.layer.{i}.output.LayerNorm.bias"
        })

    # Mapping for the final pooler layer
    mapping.update({
        "final_linear.weight": "pooler.dense.weight",
        "final_linear.bias": "pooler.dense.bias"
    })

    # Copy the weights
    for custom_param, base_param in mapping.items():
        if isinstance(base_param, list):  # Handle combined qkv weights and biases
            qkv_weight = torch.cat([
                base_model_state_dict[base_param[0]],
                base_model_state_dict[base_param[1]],
                base_model_state_dict[base_param[2]]
            ], dim=0)
            custom_model_state_dict[custom_param].copy_(qkv_weight)
        else:
            custom_model_state_dict[custom_param].copy_(base_model_state_dict[base_param])

    custom_model.load_state_dict(custom_model_state_dict)
    print("Weights successfully loaded from base model into custom model.")


def evaluate(classification_bert, test_dataloader):
    device='cuda:0'if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
            matches=0
            number_of_iters = 0
            for b_eval in test_dataloader:
                tokenized = tokenizer(b_eval['x'], padding=True, return_tensors="pt")
                tokenized.to(device)
                ids, mask, token_type_ids = tokenized['input_ids'], tokenized['attention_mask'], tokenized['token_type_ids']
                preds = classification_bert(ids, mask, token_type_ids)
                matches += torch.sum(torch.isclose(b_eval['y'].long().to(device), torch.argmax(preds, dim=1)))

            accuracy = matches / len(test_dataloader.dataset)
            print('accuracy: ', accuracy)
    return accuracy



def train(train_data_path, val_data_path, train_batch_size=32, eval_batch_size=200):

    device='cuda:0'if torch.cuda.is_available() else 'cpu'


    train_data = NewsDataset(train_data_path)
    val_data = NewsDataset(val_data_path)


    train_dataloader = DataLoader(train_data, batch_size=train_batch_size,
                            shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=eval_batch_size,
                            shuffle=True, num_workers=0)

    epochs=18
    accumulation_steps = 4800 // train_batch_size
    classification_bert = prepare_model()


    optimizer = torch.optim.Adam(classification_bert.parameters(), lr=0.00035)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    LRs=[]
    epoch_loss = []
    epoch_acc_train=[]
    epoch_acc_test=[]

    for epoch_num in range(1,epochs):
        
        accuracies_test=  []
        accuracies_train=  []
        losses=[]
        
        ttl_loss=0
        train_iters=0
        for i, train_batch in enumerate(train_dataloader):
            
            # Tokenize and pad
            tokenized = tokenizer(train_batch['x'], padding=True, return_tensors="pt")
            tokenized.to(device)
        
            # Create tensor for input batch
            ids, mask, token_type_ids = tokenized['input_ids'], tokenized['attention_mask'], tokenized['token_type_ids']
            
            preds = classification_bert(ids, mask,token_type_ids )
            loss = criterion(preds, train_batch['y'].long().to(device))
            
            ttl_loss+=loss.item()
            train_iters+=1
            
            loss.backward()
            
            
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                classification_bert.zero_grad()
                
                print('loss', ttl_loss / train_iters)
                losses.append(ttl_loss / train_iters)
            
                
                ttl_loss=0
                train_iters=0
                
                
                train_accuracy = torch.sum(torch.isclose(train_batch['y'].long().to(device), torch.argmax(preds, dim=1))) / train_batch_size
                accuracies_train.append(train_accuracy)
                print('train accuracy: ',train_accuracy )
                
            del preds
            del tokenized
            del loss

        scheduler.step()
        
        print('epoch ',str(epoch_num),'lr', scheduler.get_last_lr())
        LRs.append(scheduler.get_last_lr())

        acc = evaluate(classification_bert, val_dataloader)
        epoch_acc_test.append(acc)
        print('accuracy: ', acc)

        
        epoch_acc_train.append(accuracies_train)
        
        epoch_loss.append(losses)
        torch.save(classification_bert.state_dict(), 'weights'+str(epoch_num)+'.pt')

    acc_test_epoch = [a.cpu() for a in epoch_acc_test]
    acc_train_epoch = []
    for a in epoch_acc_train:
        acc_train_epoch.append([b.cpu() for b in a ])
    acc_train_epoch = np.array(acc_train_epoch).mean(axis=1)
    
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(acc_test_epoch)), acc_test_epoch, label='test')
    ax.plot(np.arange(len(acc_train_epoch)), acc_train_epoch, label='train')
    plt.legend()
    plt.savefig('bias_variance.png')

def main(bucket_name, train_data_path, val_data_path, train_batch_size=32, eval_batch_size=200):

    download_file(bucket_name, train_data_path)
    download_file(bucket_name, val_data_path)

    train(train_data_path, val_data_path, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size)

    upload_file('weights17.pt',bucket_name)
    upload_file('bias_variance.png',bucket_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str)           # positional argument
    parser.add_argument('--train_batch_size', type=int) 
    parser.add_argument('--eval_batch_size', type=int)
    args = parser.parse_args()

    main(
         args.bucket_name,
         train_data_path='train_data.json',
         val_data_path='val_data.json',
         train_batch_size=args.train_batch_size,
         eval_batch_size= args.eval_batch_size
         )