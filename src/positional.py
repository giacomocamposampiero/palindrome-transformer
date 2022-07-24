import torch
import math

# TODO experiment with other type of positional encoders, e.g.
# - original Transformer's PE with fixed sinusoidal functions
# - learned PE 

class PositionEncodingParity(torch.nn.Module):
    """
    Custom positional encoder layer for Parity learning from https://arxiv.org/pdf/2202.12172.pdf
    The authors define the PE function to be an arbitrary function on all positions.

    More formally, it is defined as
    \begin{equation*}
        pe_{ij} =  \begin{cases}
                        \frac{i}{n} \cdot \exp(\gamma_j)  \qquad \text{if} j \le lfloor\frac{n}{2}\rfloor
                        \cos(i\pi \cdot \exp(\gamma_j))   \qquad \text{if} j > lfloor\frac{n}{2}\rfloor
                    \end{cases}
    \end{equation*}
    where $0 \le i \le n$, $0 \le j \le size$ and $\gamma_j \sim \mathcal{N}(0,1)$.
    """

    def __init__(self, size):
        """
        Initialize positional embedder.

        Args:
            size: max length of a sequence that can be encoded by the transformer (required).
        """

        super().__init__()
        assert size % 2 == 0
        self.size = size

        # draw tensor of shape (size,) at random from normal distributions
        # with μ = 0 and σ = 1
        self.scales = torch.nn.Parameter(torch.normal(0, 1., (size,)))

    def forward(self, n):
        """
        Compute positional embeddings for a sequence of lenght n.

        Args:
            n: length of the sequence (required).
        """

        # absolute positions 
        p = torch.arange(0, n).to(torch.float).unsqueeze(1)

        # positional embeddings as in the formula above
        pe = torch.cat([
            p / n * torch.exp(self.scales[:n//2]), # (n, n//2) 
            torch.cos(p*math.pi * torch.exp(self.scales[n//2:])) # (n, size - n//2) 
        ], dim=1)

        return pe


class PositionEncodingFirst(torch.nn.Module):
    """
    Custom positional encoder layer for First learning from https://arxiv.org/pdf/2202.12172.pdf.
    In this case, the positional embedding function is defined as 
    \begin{equation*}
        pe_{ij} =  \begin{cases}
                        1  \qquad \text{if} j = 0, \, i = 1
                        0  \qquad \text{otherwise}
                    \end{cases}
    \end{equation*}
    where $0 \le i \le n$ and $0 \le j \le size$.
    """

    def __init__(self, size):
        """
        Initialize positional embedder.

        Args:
            size: max length of a sequence that can be encoded by the transformer (requird).
        """

        super().__init__()
        self.size = size

    def forward(self, n):
        """
        Compute positional embeddings for a sequence of lenght n.

        Args:
            n: length of the sequence (required).
        """

        zero = torch.zeros(n)
        pos = torch.arange(0, n).to(torch.float)
        pe = torch.stack([pos == 1] + [zero]*(self.size-1), dim=1)
        return pe


class PositionEncodingFirstExact(torch.nn.Module):
    def __init__(self):
        """
        Initialize positional embedder.

        Args:
            size: max length of a sequence that can be encoded by the transformer (requird).
        """

        super().__init__()

    def forward(self, n):
        """
        Compute positional embeddings for a sequence of lenght n.

        Args:
            n: length of the sequence (required).
        """
        
        zero = torch.zeros(n)
        pos = torch.arange(0, n).to(torch.float)
        pe = torch.stack([zero] * 3 + [pos == 1] + [zero] * 2, dim=1)
        return pe

class PositionEncodingParityExact(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, n):
        zero = torch.zeros(n)
        pos = torch.arange(0, n).to(torch.float)
        pe = torch.stack([zero]*3 +
                         [pos / n,
                          torch.cos(pos*math.pi)] +
                         [zero]*5,
                         dim=1)
        return pe


class PositionEncodingOne(torch.nn.Module):
    """
    Custom positional encoder layer for One learning and exact
    In this case, the positional embedding function is defined as 
    \begin{equation*}
        pe_{ij} =  \frac{i}{n}
    \end{equation*}
    where i is the position of the word and n the size of the sequence.
    """

    def __init__(self, size):
        """
        Initialize positional embedder.

        Args:
            size: max length of a sequence that can be encoded by the transformer (required).
        """
        super().__init__()
        self.size = size

    def forward(self, n):
        """
        Compute positional embeddings for a sequence of lenght n.

        Args:
            n: length of the sequence (required).
        """

        zero = torch.zeros(n)
        pos = torch.arange(0, n).to(torch.float)
        pos = pos / self.size
        pe = torch.stack([pos] + [zero]*(self.size-1), dim=1)
        return pe

# # DEBUG PE ONE
# pos = PositionEncodingOne(3)
# print(pos.forward(4))
