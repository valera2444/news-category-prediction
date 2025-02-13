import torch
import torch.nn as nn
import math

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, context_length):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(context_length, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return embeddings


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape,  eps=0.00001):
        super().__init__()
        self.eps=eps
        self.normalized_shape=normalized_shape
        self.gamma = nn.Parameter(torch.ones((1,1,self.normalized_shape)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1,1,self.normalized_shape)), requires_grad=True)

    def forward(self, input):
        """

        Args:
            input (torch.Tensor): (B, T, C)

        Returns:
            torch.Tensor: (B, T, C)
        """
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True,  unbiased=False)
        
        out = (input - mean) / torch.sqrt(var+self.eps) * self.gamma + self.beta
        
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, n_heads, p_dropout = 0.0):

        super().__init__()
        assert model_dim % n_heads == 0
        
        self.dim = model_dim 
        self.n_heads=n_heads
        self.head_dim = self.dim // self.n_heads
        

        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=True)

        self.dropout = nn.Dropout(p=p_dropout)

        self.proj = nn.Linear(model_dim, model_dim, bias=True)#part of attention

        

    def forward(self, input, attn_mask):
        """_summary_

        Args:
            input (torch.tensor): (B,T,C)
            attn_mask (torch.tensor): (B,T)

        Returns:
            _type_: _description_
        """

        #Mutihead attention
        qkv = self.qkv(input) # фееутешщт ьфыл шт иуке(B, T, 3*dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(input.size(0), input.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(input.size(0), input.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(input.size(0), input.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        weights = q @ k.transpose(-2,-1) * self.head_dim**(-0.5) # (B,n_heads, T , T)

        
        weights = weights.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2) == 0, -torch.inf)
        
        weights = torch.nn.functional.softmax (weights, dim=3)

        weights = self.dropout(weights)
        
        out = weights @ v
        
        out=out.transpose(1,2) # (B,T, n_heads , head_dim)
        
        #Concatenation of heads
        out = out.reshape(out.size(0), out.size(1),-1) # (B,T, C)
        out = self.proj(out)

        return out


class TransfomerBlock(nn.Module):

    def __init__(self, model_dim, n_heads, max_seq_len, p_dropout = 0.0):

        super().__init__()
        assert model_dim % n_heads == 0
        
        self.dim = model_dim 
        self.n_heads=n_heads
        self.head_dim = self.dim // self.n_heads
        

        self.mha = MultiHeadAttention(model_dim, n_heads, max_seq_len)

        self.ln1 = LayerNorm(model_dim)

        self.dropout= nn.Dropout(p=p_dropout)

        self.mlp = nn.Sequential(nn.Linear(model_dim, 4*model_dim),
                                 nn.GELU(),
                                 nn.Linear(4*model_dim, model_dim),
                                 )
        
        self.ln2 = LayerNorm(model_dim)


    def forward(self, input, mask):
        """_summary_

        Args:
            input (torch.tensor): input for block with dimensions: (B, T, C), where B - batch, T - sequence length, C - model size
            mask (torch.tensor): (B, T)
        Returns:
            Next tokens for each k-gram. Shape : # (B,T, C)
        """
        residual_1 = input
        
        #Mutihead attention
        out = self.mha (input, mask)

        out = self.ln1 (out)

        out = self.dropout(out)
        #Add
        out = residual_1 + out

        residual_2 = out
        
        #Feed forward
        out = self.mlp (out)

        out = self.ln2(out)

        out = self.dropout(out)
        #Add
        out = residual_2 + out
        


        return out



class Transformer(nn.Module):
    """
    Encoder-only transofrmer.
    forward method takes sequences with the same lengthes as input, and then attention mask applied for pad tokens
    """
        
    def __init__(self, vocab_size, model_dim, n_heads, max_seq_len, n_blocks, p_dropout=0.0):
        super().__init__()
        self.vocab_size, self.model_dim, self.n_heads, self.max_seq_len, self.n_blocks = vocab_size, model_dim, n_heads, max_seq_len, n_blocks
        self.emb  = BertEmbeddings(vocab_size, model_dim,max_seq_len)
        self.emb_ln = LayerNorm(model_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.net = nn.ModuleList([TransfomerBlock(model_dim, n_heads,p_dropout) for _ in range (n_blocks)])
       
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.activation = nn.Tanh()


    def forward(self, input, attn_mask, token_type_ids):
        """
        input (torch.Tensor): batch of data to predict logits on. (B, T, C)
        Args:
            input (torch.Tensor): batch of data to predict logits on. (B, T, C)
            attn_mask (torch.Tensor): pad tokens indicator. (B, T)
        Returns:
            torch.Tensor: Tensors with logits. nan where pad token. (B, T, vocab_size) - mlm_logits, (B, 2)- nsp logits
        """
        
        embs = self.emb(input, token_type_ids)

        embs = self.emb_ln (embs)

        embs = self.dropout(embs)

        for encoder in self.net:
            embs = encoder.forward(embs, attn_mask)

        hidden = embs

        out = self.final_linear(embs)#(B, T, C)

        out = self.activation(out)

        return hidden, out
    

class BertPooler(nn.Module):
        def __init__(self, model_dim):
            super().__init__()
            self.dense = nn.Linear(model_dim, model_dim)
            self.activation = nn.Tanh()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            first_token_tensor = hidden_states[:, 0, :]
            #print(first_token_tensor.shape)
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        
class CustomBertForClassification(nn.Module):


    
        
    def __init__(self, bert_base, hidden_dim, num_labels):

        super().__init__()
        self.bert_base = bert_base
        self.hidden_dim = hidden_dim
        self.num_labels=num_labels
        self.pooler = BertPooler(bert_base.model_dim)
        self.classification_head = nn.Sequential(nn.Linear(bert_base.model_dim,hidden_dim ),
                                                 nn.ReLU(),
                                                 nn.Linear(hidden_dim, num_labels))
        
        
    def forward(self, input, attn_mask, token_type_ids):
        """_summary_

        Args:
            input (_type_): _description_
            attn_mask (_type_): _description_
            token_type_ids (_type_): _description_

        Returns:
            torch.Tensor: logits for num_labels classes
        """

        hidden, out = self.bert_base(input, attn_mask, token_type_ids)
        #print(out.shape)
        out=self.pooler(hidden)#Must be applied on hidden state
        out = self.classification_head(out)
        return out

