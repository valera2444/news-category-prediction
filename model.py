
import torch
import torch.nn as nn


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
        qkv = self.qkv(input) #  (B, T, 3*dim)
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

