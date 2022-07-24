from abc import abstractmethod
import torch
from .positional import PositionEncodingFirst, PositionEncodingParity, PositionEncodingFirstExact, PositionEncodingParityExact, PositionEncodingOne, PositionEncodingPalindrome
from .encoder import ScaledTransformerEncoderLayer, StandardTransformerEncoderLayer, FirstExactEncoder, ParityExactEncoder

class Transformer(torch.nn.Module):
    """
    Transformer class with abstract encoder.
    """

    def __init__(self, alphabet_size, d_model):
        """
        Initialize Transformer module.

        Args:
            alphabet_size: |Σ|
            d_model: the number of expected features in the encoder/decoder inputs.
        """

        super().__init__()
        torch.manual_seed(0)
        # word embedding layer
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)

        # encoder gets set by the specific subclass
        @property
        @abstractmethod
        def encoder(self):
            pass

        # final linear layer for the output 
        self.output_layer = torch.nn.Linear(d_model, 1)


class StandardTransformer(Transformer):
    """
    Custom Transfomer model with traditional encoder layers, based on based on Vaswani et al. 2017 (https://arxiv.org/pdf/1706.03762.pdf)
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

        super().__init__(alphabet_size, d_model)

        # use Scaled or Regular Transformer encoder layer
        # nb we're not using dropout
        if scaled:
            encoder_layer = ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = StandardTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

class FirstTransformer(StandardTransformer):
    """
    Transformer Encoder (without decoding step) to learn First language.
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

        super().__init__(alphabet_size, layers, heads, d_model, d_ffnn, scaled, eps)
        self.pos_encoding = PositionEncodingFirst(d_model)
    
    def forward(self, w):
        """
        Perform forward pass.

        Args:
            w: word
            pos: position of the output layer to return
        Returns:
            single output from the output layer at specified position.
        """
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z

class ParityTransformer(StandardTransformer):
    """
    Transformer Encoder (without decoding step) to learn parity language.
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

        super().__init__(alphabet_size, layers, heads, d_model, d_ffnn, scaled, eps)
        self.pos_encoding = PositionEncodingParity(d_model)
    
    def forward(self, w):
        """
        Perform forward pass.

        Args:
            w: word
            pos: position of the output layer to return
        Returns:
            single output from the output layer at specified position.
        """
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[-1])
        return z


class OneTransformer(StandardTransformer):
    """
    Transformer Encoder (without decoding step) to learn One language.
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

        super().__init__(alphabet_size, layers, heads, d_model, d_ffnn, scaled, eps)
        self.pos_encoding = PositionEncodingOne(d_model)
    
    def forward(self, w):
        """
        Perform forward pass.

        Args:
            w: word
        Returns:
            single output from the output layer at specified position.
        """
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z


class PalindromeTransformer(StandardTransformer):
    """
    Transformer Encoder (without decoding step) to learn One language.
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

        super().__init__(alphabet_size, layers, heads, d_model, d_ffnn, scaled, eps)
        self.pos_encoding = PositionEncodingPalindrome(d_model)
    
    def forward(self, w):
        """
        Perform forward pass.

        Args:
            w: word
        Returns:
            single output from the output layer at specified position.
        """
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z



# Exact implementations

class FirstExactTransformer(Transformer):
    """
    Transformer Encoder (without decoding step) to learn First language exactly.
    """

    def __init__(self, alphabet_size, d_model, normalize=False, eps=1e-5):
        """
        Initialize Transformer module.

        Args:
            alphabet_size: |Σ|
            d_model: the number of expected features in the encoder/decoder inputs.
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        super().__init__(alphabet_size, d_model)
        self.pos_encoding = PositionEncodingFirstExact()
        self.encoder = FirstExactEncoder(normalize=normalize, eps=eps)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,1]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        """
        Perform forward pass.

        Args:
            w: word
            pos: position of the output layer to return
        Returns:
            single output from the output layer at specified position.
        """


        inter=self.pos_encoding(len(w))
        inter2=self.word_embedding(w)
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z

class ParityExactTransformer(Transformer):
    def __init__(self,alphabet_size,d_model):
        super().__init__(alphabet_size,d_model)
        # self.word_embedding = torch.eye(3, 10)
        self.pos_encoding = PositionEncodingParityExact()
        self.encoder = ParityExactEncoder()
        # self.output_layer = torch.nn.Linear(10, 1)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,0,1,0]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        return z
