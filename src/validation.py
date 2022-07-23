import torch
from src.transformer import Transformer
from src.dataset import Dataset

class Validator:
    """
    Validator
    -------
    
    Model validator for exact Transformers. 

    Parameters
    ----------

    uid : int
          run univoque identification number

    model : src.transformer.Transformer
            transformer model to be trained

    vocab : list
            vocabulary for the transformer
    
    valset : src.dataset.Dataset
               validation dataset

    verbose : int
              verbosity level of the trainer, 0 for no verbosity, 1 otherwise

    """
    def __init__(self, 
                 uid : int,
                 model : Transformer, 
                 vocab : list,
                 valset : Dataset,
                 verbose : int
                ) -> None:

        # assert consistency between uids
        assert uid == valset.uid

        # initialize trainer parameters
        self.uid = uid
        self.model = model
        self.vocab = vocab
        self.valset = valset
        self.verbose = verbose
        self.log_sigmoid = torch.nn.LogSigmoid()
        

    def validate(self) -> tuple[float, float]:
        """
        Validate the model using the provided validation set.

        Returns
        -------
        loss : float
               validation loss

        acc : float
              validation accuracy
        """

        loss = 0
        correct = 0

        self.model.eval()

        with torch.no_grad():

            for x, y in self.valset:

                # forward step
                x = self.__encode(x)
                output = self.model(x)

                if not y: output = -output
                if output > 0: correct += 1

                # compute loss
                model_los = -self.log_sigmoid(output)
                loss += model_los.item()

        # print validation info
        if self.verbose:
            print(f"[Validation length {self.valset.length}] Loss: {loss}, Accuracy: {correct/self.valset.size}", flush=True)

        return loss, correct / self.valset.size

    def __encode(self, s: str) -> torch.Tensor:
        alphabet_index = {a:i for i,a in enumerate(self.vocab)}
        t = torch.tensor([alphabet_index[c] for c in s])
        return t
