from abc import abstractmethod
import torch
from .positional import PositionEncodingFirst, PositionEncodingParity, PositionEncodingFirstExact, PositionEncodingParityExact, PositionEncodingPalindromeExact, PositionEncodingOneExact, StandardPositionalEncoding
from .encoder import ScaledTransformerEncoderLayer, StandardTransformerEncoderLayer, FirstExactEncoder, ParityExactEncoder, PalindromeExactEncoder, OneExactEncoder

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
    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5, positional = 'standard', cls_pos = 0):
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
            positional: type of positional encodings to be used.
            cls_pos: output position to be used to classify.
        """

        super().__init__(alphabet_size, d_model)

        # use Scaled or Regular Transformer encoder layer
        # nb we're not using dropout
        if scaled:
            encoder_layer = ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = StandardTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        
        # select positional encoding
        if positional == 'standard':
            self.pos_encoding = StandardPositionalEncoding(d_model)
        elif positional == 'first':
            self.pos_encoding = PositionEncodingFirst(d_model)
        elif positional == 'parity':
            self.pos_encoding = PositionEncodingParity(d_model)
        elif positional == 'one':
            self.pos_encoding = PositionEncodingOneExact(d_model)
        elif positional == 'palindrome':
            self.pos_encoding = PositionEncodingPalindromeExact(d_model)
        
        self.cls_pos = cls_pos

        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

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
        z = self.output_layer(y[self.cls_pos])
        return z


# Exact implementations

# --------------------- Exact Transformers ----------------------- #

class FirstExactTransformer(Transformer):
    """
    Transformer Encoder (without decoding step) to learn First language exactly.
    """

    def __init__(self, alphabet_size, d_model):
        """
        Initialize Transformer module.

        Args:
            alphabet_size: |Σ|
            d_model: the number of expected features in the encoder/decoder inputs.
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        super().__init__(alphabet_size, d_model)
        self.exact_word_embedding = torch.eye(3, d_model)
        self.pos_encoding = PositionEncodingFirstExact()
        self.encoder = FirstExactEncoder()
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,1]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, w):
        """
        Perform forward pass.

        Args:
            w: word
            pos: position of the output layer to return
        Returns:
            single output from the output layer at specified position.
        """
        x = self.exact_word_embedding[w] + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        output = self.activation(z)
        return output, float(self.logsigmoid(z)[0])

    def activation(self, z):
        z = z[0]
        if z > 0:
            # There is a one in the first position
            return True
        else:
            # There is not a one in the first position
            return False

class ParityExactTransformer(Transformer):
    def __init__(self, alphabet_size, d_model):
        """
        Initialize Transformer module.

        Args:
            alphabet_size: |Σ|
            d_model: the number of expected features in the encoder/decoder inputs.
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        super().__init__(alphabet_size, d_model)
        self.exact_word_embedding = torch.eye(3, d_model)
        self.pos_encoding = PositionEncodingParityExact()
        self.encoder = ParityExactEncoder()
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,0,1,0]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, w):
        """
        Perform forward pass.
        Args:
            w: word
            pos: position of the output layer to return
        Returns:
            single output from the output layer at specified position.
        """
        x = self.exact_word_embedding[w] + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        output = self.activation(z)
        return output, float(self.logsigmoid(z)[0])

    def activation(self, z):
        z = z[0]
        if z > 0:
            # has odd number of ones
            return False
        else:
            # has even number of ones
            return True
        


class OneExactTransformer(Transformer):
    """
    Transformer Encoder (without decoding step) to learn One language exactly
    """
    def __init__(self, alphabet_size, d_model):
        """
        Initialize Transformer module.
        Args:
            alphabet_size: |Σ|
            d_model: the number of expected features in the encoder/decoder inputs.
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        super().__init__(alphabet_size, d_model)
        self.exact_word_embedding = torch.eye(3, d_model)
        self.pos_encoding = PositionEncodingOneExact()
        self.encoder = OneExactEncoder()
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,1]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, w):
        """
        Perform forward pass.
        Args:
            w: word
            pos: position of the output layer to return
        Returns:
            single output from the output layer at specified position.
        """
        x = self.exact_word_embedding[w] + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        output = self.activation(z)
        return output, float(self.logsigmoid(z)[0])

    def activation(self, z):
        z = z[0]
        if z > 0:
            # Only contains a single one
            return True
        else:
            # Does not contain a single one
            return False

class PalindromeExactTransformer(Transformer):
    """
    Transformer Encoder (without decoding step) to learn Palindrome language exactly (theoretically)
    """
    def __init__(self, alphabet_size, d_model, error=1e-7):
        """
        Initialize Transformer module.
        Args:
            alphabet_size: |Σ|
            d_model: the number of expected features in the encoder/decoder inputs.
            normalize: whether layer normalization should be applied
            eps: the epsilon value for layer normalization in both layers
        """
        super().__init__(alphabet_size, d_model)
        self.error = error
        self.exact_word_embedding = torch.eye(4, d_model)
        self.pos_encoding = PositionEncodingPalindromeExact()
        self.encoder = PalindromeExactEncoder()
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,0,0,0,1,0]], dtype=torch.float))
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
        x = self.exact_word_embedding[w] + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[0])
        output = self.activation(z)
        return output, 0

    def activation(self, z):
        z = z[0]
        if abs(z) <= self.error:
            # Is palindrome if z = 0, or precision error to 0 is close enough
            return True
        else:
            # Otherwise it is not a palindrome
            return False





