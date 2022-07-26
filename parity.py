from src.transformer import ParityExactTransformer
import torch
import pandas as pd

EXACT = True


log_sigmoid = torch.nn.LogSigmoid()
alphabet = ["0", "1", "$"]

epochs = 20
d_model = 10
alphabet_index = {a : i for i, a in enumerate(alphabet)}


train = pd.read_csv("data/parity/test_n2000_l1000.csv")
test = pd.read_csv("data/parity/test_n100_l100.csv")
X_train, y_train = train['sequence'].values, train['label'].values
X_test, y_test = test['sequence'].values, test['label'].values



def _encode(s : str) -> torch.Tensor:
    t = torch.tensor([alphabet_index[c] for c in s])
    return t

X_train = [_encode(s) for s in X_train]
X_test = [_encode(s) for s in X_test]


transformer = ParityExactTransformer(len(alphabet), d_model)
for param in transformer.named_parameters():
    print(param[0])
    print(param[1])
    print()
        
optim = torch.optim.Adam(transformer.parameters(), lr=0.0003)

print("Done printing parameters")
print()


train_l = []
val_l = []
train_acc = []
val_acc = []

for epoch in range(epochs):

    train_loss = train_correct = 0    

    cnt = 0
    
    correct = 0
    incorrect = 0
    EPS = 1e-5
    # train step
    for x, y in zip(X_train, y_train):

        output = transformer(x)
        print("Correct : ", correct)
        print("Incorrect : ", incorrect)
 
        if y == output:
            correct += 1
        else:
            incorrect += 1

        cnt += 1
        print("------------------------")




    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    exit()
    # save statistic about training step

