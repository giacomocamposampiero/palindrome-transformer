import torch
from src.transformer import Transformer
from src.dataset import Dataset

class Trainer:
    """
    Trainer
    -------
    
    Model trainer for learned Transformers. 

    Parameters
    ----------

    uid : int
          run univoque identification number

    model : src.transformer.Transformer
            transformer model to be trained
    
    optim : int
            torch optimizer for the transformer

    vocab : list
            alphabet of characters

    epochs : int
             number of training epochs
    
    trainset : src.dataset.Dataset
               training dataset

    testset : src.dataset.Dataset
              test dataset

    verbose : int
              verbosity level of the trainer, 0 for no verbosity, 1 otherwise

    """
    def __init__(self, 
                 uid : int,
                 model : Transformer, 
                 optim : torch.optim.Optimizer, 
                 vocab : list,
                 epochs : int, 
                 trainset : Dataset, 
                 testset : Dataset,
                 verbose : int
                ) -> None:

        # assert consistency between uids
        assert uid == trainset.uid == testset.uid

        # initialize trainer parameters
        self.uid = uid
        self.model = model
        self.optim = optim
        self.vocab = vocab
        self.epochs = epochs
        self.trainset = trainset
        self.testset = testset
        self.verbose = verbose
        self.log_sigmoid = torch.nn.LogSigmoid()
        

    def train(self) -> tuple[list, list, list, list]:
        """
        Train the model.

        Returns
        -------
        train_l : list
                  training loss for each epoch, shape (epochs,)

        val_l : list
                validation loss for each epoch, shape (epochs,)

        train_acc : list
                    training accuracy for each epoch, shape (epochs,)

        val_acc : list
                  validation accuracy for each epoch, shape (epochs,)
        """
        
        train_l = []
        val_l = []
        train_acc = []
        val_acc = []

        for epoch in range(self.epochs):
            
            self.trainset.reset_index()
            self.testset.reset_index()
            train_loss = train_correct = 0    
            
            # train step
            for x, y in self.trainset:

                # forward step
                x = self.__encode(x)
                output = self.model(x)

                if not y: output = -output
                if output > 0: train_correct += 1

                # compute loss
                loss = -self.log_sigmoid(output)
                train_loss += loss.item()

                # optimizer step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # save statistics about training step
            train_l.append(train_loss) # loss
            train_acc.append(train_correct/self.trainset.size) # accuracy
                
            # validation step
            with torch.no_grad():

                test_loss = test_correct = 0

                for x, y in self.testset:

                    x = self.__encode(x)
                    output = self.model(x)

                    if not y: output = -output
                    if output > 0: test_correct += 1

                    loss = -self.log_sigmoid(output)
                    test_loss += loss.item()

            # save statistic about validation step
            val_l.append(test_loss) # loss
            val_acc.append(test_correct/self.testset.size) # accuracy

            # print step info
            if self.verbose:
                print(f"[Epoch {epoch+1}] Train acc: {train_correct/self.trainset.size} Train loss: {train_loss}, Test acc: {test_correct/self.testset.size} Test loss: {test_loss}", flush=True)

        return train_l, val_l, train_acc, val_acc
  

    def __encode(self, s: str) -> torch.Tensor:
        alphabet_index = {a:i for i,a in enumerate(self.vocab)}
        t = torch.tensor([alphabet_index[c] for c in s])
        return t
