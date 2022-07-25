from typing import Optional

from numpy import log as ln

import torch
from torch import Tensor
import math

class StandardTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer.
        To understand forward in Transformer Encoder layer, check: Figure 1a https://arxiv.org/pdf/2002.04745v1.pdf

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        x = src
        
        # Transformer's [self-attention + add & norm]
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        # Transformer's [feedforward + add & norm]  
        x = self.norm2(x + self._ff_block(x))
        
        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Multi-Head Attention layer that also returns last layer for model investigation.

        Args:
            x: input sequence (required).
            attn_mask: the mask for the input sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, self.last_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)


class ScaledTransformerEncoderLayer(StandardTransformerEncoderLayer):

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], 
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Multi-Head Attention layer that also returns last layer for model investigation.
        Logits of each attention layer are scaled by log(n).

        Args:
            x: input sequence (required).
            attn_mask: the mask for the input sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, self.last_weights = self.self_attn(x*math.log(len(x)), x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)

# -------------  Exact  ---------------  #
        
class FirstExactEncoder(torch.nn.TransformerEncoder):
    
    def __init__(self, normalize, eps):
        """
        Custom encoder as described in https://arxiv.org/pdf/2202.12172.pdf.

        Args:
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            FirstExactTransformerFirstLayer(normalize, eps),
            FirstExactTransformerSecondLayer(normalize, eps),
        ])
        self.num_layers = len(self.layers)
        self.norm = None
    
class FirstExactTransformerFirstLayer(torch.nn.TransformerEncoderLayer):

    def __init__(self, normalize, eps):
        """ 
        Custom single head attention layer as described in https://arxiv.org/pdf/2202.12172.pdf.
        Args:
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        EMBED_DIM = 6
        W_F1 = [[-1, 0, -1, 1, 0, 0]]
        b_F1 = [0]
        W_F2 = [[0], [0], [0], [0], [1], [0]]
        b_F2 = [0] * EMBED_DIM

        super().__init__(d_model=EMBED_DIM, nhead=1, dim_feedforward=1, dropout=0.)

        self.normalize = normalize
        self.norm1.eps = self.norm2.eps = eps

        # q, k, v all have the same embedding dimension, so we just have to set it once as in_proj
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.zeros(3 * EMBED_DIM, EMBED_DIM))
        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * EMBED_DIM))

        # out_proj weights and bias
        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.zeros(EMBED_DIM, EMBED_DIM))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(EMBED_DIM))

        # First FFNN
        self.linear1.weight = torch.nn.Parameter(torch.tensor(W_F1, dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.tensor(b_F1, dtype=torch.float))

        # Second FFNN
        self.linear2.weight = torch.nn.Parameter(torch.tensor(W_F2, dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.tensor(b_F2, dtype=torch.float))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.normalize:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.normalize:
            src = self.norm2(src)
        return src

class FirstExactTransformerSecondLayer(torch.nn.TransformerEncoderLayer):

    def __init__(self, normalize, eps):
        """ 
        Custom single head attention layer as described in https://arxiv.org/pdf/2202.12172.pdf.

        Args:
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        EMBED_DIM = 6
        W_Q = [[0, 0, 1.0, 0, 0, 0]] + [[0] * EMBED_DIM] * 5
        W_K = [[0, 0, 0, 1, 0, 0]] + [[0] * EMBED_DIM] * 5
        W_V = [[0] * EMBED_DIM] * 5 + [[0, 0, 0, -0.5, 1, 0]]
        W_O = [[0] * EMBED_DIM] * 5 + [[0, 0, 0, 0, 0, 1]]

        super().__init__(d_model=EMBED_DIM, nhead=1, dim_feedforward=1, dropout=0.0)

        self.normalize = normalize
        self.norm1.eps = self.norm2.eps = eps

        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            W_Q +
            W_K +
            W_V, dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * EMBED_DIM))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(W_O, dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(EMBED_DIM))

        self.linear1.weight = torch.nn.Parameter(torch.zeros(1,EMBED_DIM))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(1))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(EMBED_DIM,1))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(EMBED_DIM))

    forward = FirstExactTransformerFirstLayer.forward


# -------------  Parity ---------------  #

class ParityExactTransformerFirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            
            [[0]*10]*10 +
            
            [[0]*10]*10 +
           
            [[0,1,0,0,0,0,0,0,0,0],   
             [0,0,1,0,0,0,0,0,0,0]]+   
            [[0]*10]*8,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
           
            [[0]*10]*5 +
            [[1,0,0,0,0,0,0,0,0,0],   
             [0,1,0,0,0,0,0,0,0,0]] +
            [[0]*10]*3,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [0,0,0,-1,0,1,-1,0,0,0],  
            [0,0,0,-1,0,1, 0,0,0,0],  
            [0,0,0,-1,0,1, 1,0,0,0],  
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0, 0, 0]]*7 +
            [[1,-2, 1],  
             [0, 0, 0],
             [0, 0, 0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class ParityExactTransformerSecondLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            [[0,0,1,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            [[0,0,1,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
           
            [[0,0,0,0, 1,0,0,0,0,0]] +
            [[0]*10]*4 +
            
            [[0,0,0,0,-1,0,0,0,0,0]] +
            [[0]*10]*4 +
           
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4 +
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            
            [[0]*10]*8 +
            [[-1,0,0,0,0,1,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0]],
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.zeros(3,10))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(10,3))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        q = src
        v = src
        src2 = self.self_attn(q, src, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class ParityExactEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            ParityExactTransformerFirstLayer(),
            ParityExactTransformerSecondLayer(),
        ])
        self.num_layers = len(self.layers)
        self.norm = None


# -------------  Palindrome ---------------  #

class PalindromeExactTransformerFirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            
            # W^Q
            [[0]*10]*10 +

            # W^K
            [[0]*10]*10 +
           
            # W^V
            [[0]*10]*10,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
           
            [[0]*10]*10,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [-1,0,-1,0,0,1,0,0,0,0],  
            [-1,0,-1,0,0,0,1,0,0,0]  
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(2))
        
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0, 0]]*7 +
            [[1, 0],  
             [0, 1],
             [0, 0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))


        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        #print(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        #print("After doing the src")
        #print(src)
        return src

class PalindromeExactTransformerSecondLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # W^Q
            # Heads 1
            [[0,0,1,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            # Head 2
            [[0,0,1,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +


            # W^K
            # Head 1
            [[0,0,0,1,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            
            # Head 2
            [[0,0,0,0,1,0,0,0,0,0]] +
            [[0]*10]*4 +
           
            # W^V
            # Head 1
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4 +

            # Head 2
            [[0,0,0,0,0,0,0,0,1,0]] +
            [[0]*10]*4,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            
            # W^O
            [[0]*10]*8 +
            [[0,0,0,0,0,0,0,0,0,0],
             [1,0,0,0,0,-1,0,0,0,0]],

            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.zeros(3,10))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(10,3))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        #print("second layer")
        #print(src)
        q = src
        v = src
        src2 = self.self_attn(q, src, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        #print(src)
        return src



class PalindromeExactEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            PalindromeExactTransformerFirstLayer(),
            PalindromeExactTransformerSecondLayer(),
        ])

        self.num_layers = len(self.layers)
        self.norm = None





