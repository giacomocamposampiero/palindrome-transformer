from typing import Optional

import torch
from torch import Tensor
import math

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):

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


class ScaledTransformerEncoderLayer(TransformerEncoderLayer):

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
        