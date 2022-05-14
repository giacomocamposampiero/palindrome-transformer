import torch
from .positional import PositionEncodingFirst, PositionEncodingParity
from .encoder import ScaledTransformerEncoderLayer, TransformerEncoderLayer

class Transformer(torch.nn.Module):
    """
    Custom Transformer model, based on Vaswani et al. 2017 (https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):
        """
        Initialize Transformer module.

        Args:
            alphabet_size: |Σ|
            layers: the number of sub-layers in the encoder.
            heads: the number of heads in the multiheadattention models.
            d_model: the number of expected features in the encoder/decoder inputs.
            d_ffnn: the dimension of the feedforward network model.
            scaled: boolean flag to specify whether use normal or scaled encoder layer.
            eps: the eps value in layer normalization components.
        """
        
        super().__init__()
        
        # word embedding layer
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)
        
        # use Scaled or Regular Transformer encoder layer
        # nb we're not using dropout
        if scaled:
            encoder_layer = ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # final linear layer for the output 
        self.output_layer = torch.nn.Linear(d_model, 1)

    
class FirstTransformer(Transformer):
    """
    Transformer Encoder (without decoding step) to learn First language.
    """

    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):

        super().__init__(alphabet_size, layers, heads, d_model, d_ffnn, scaled, eps)
        self.pos_encoding = PositionEncodingFirst(d_model)
    
    def forward(self, w):
        
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        y = y[0]
        z = self.output_layer(y)
        return z

class ParityTransformer(Transformer):
    """
    Transformer Encoder (without decoding step) to learn parity language.
    """

    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):

        super().__init__(alphabet_size, layers, heads, d_model, d_ffnn, scaled, eps)
        self.pos_encoding = PositionEncodingParity(d_model)

    def forward(self, w):

        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        y = y[-1]
        z = self.output_layer(y)
        return z
