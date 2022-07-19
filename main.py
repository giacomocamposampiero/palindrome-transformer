import argparse
from src.transformer import FirstTransformer, FirstExactTransformer, ParityExactTransformer, ParityTransformer
from src.trainer import Trainer
from src.dataset import Dataset
import torch
import os
import csv

def main(args):

    rnd_seeds = [477, 653, 720, 497, 112, 350, 246, 634, 307, 655, 51, 737, 876, 611, 139, 291, 558, 47, 396, 210, 126, 866, 993, 590, 974, 680, 810, 573, 715, 249, 361, 616, 151, 468, 348, 633, 449, 312, 699, 806, 493, 234, 750, 846, 819, 409, 579, 331, 164, 198, 738, 901, 679, 813, 909, 267, 791, 497, 703, 674, 578, 904, 897, 570, 17, 240, 848, 906, 529, 118, 543, 663, 451, 557, 159, 612, 873, 791, 7, 308, 401, 231, 671, 482, 983, 650, 338, 143, 159, 456, 148, 6, 341, 630, 524, 861, 393, 521, 351, 271]

    for run in range(args.runs):

        seed = rnd_seeds[run]

        # define number of layers
        if args.exact: d_model = 6
        else: d_model = 16

        # language switch
        if args.lan == 'first':
            vocab = ["0", "1", "$"]
            transformer = FirstExactTransformer(len(vocab), d_model) if args.exact \
                    else FirstTransformer(len(vocab), args.layers, args.heads, d_model, args.d_ffnn, args.scaled, args.eps)
            
        elif args.lan == 'first':
            vocab = ["0", "1", "$"]
            transformer = ParityExactTransformer(len(vocab), d_model) if args.exact \
                    else ParityTransformer(len(vocab), args.layers, args.heads, d_model, args.d_ffnn, args.scaled, args.eps)

        # TODO other languages
        else: raise ValueError(f"{args.lan} language not supported.")
        
        optim = torch.optim.Adam(transformer.parameters(), lr=args.lr)

        # unique run id retrieval
        from pathlib import Path
        path = Path("run.id")
        if path.exists():
            file1 = open("run.id", "r")
            runid = int(file1.read()) + 1
        else:
            runid = 1
        print(f"[RUNID {runid}] Running models.")
        file1 = open("run.id", "w")
        file1.write(str(runid))
        # log model details
        log_model(runid, args)

        # train the model
        trainset = Dataset(runid, args.size, args.train_length, random_seed=seed, train=True, data_type=args.lan, variable_lenght=args.varlen)
        testset = Dataset(runid, args.size, args.test_length,  random_seed=seed,  train=False, data_type=args.lan, variable_lenght=args.varlen)

        trainer = Trainer(runid, transformer, optim, vocab, args.epochs, trainset, testset, verbose=1)
        train_l, val_l, train_acc, val_acc = trainer.train()

        log_results(runid, args, train_l, val_l, train_acc, val_acc)

def log_model(runid, args):

    from pathlib import Path
    path = Path("data/models.csv")
    if not path.exists():
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['runid', 'lan', 'train_length', 'test_length', 'size', 'varlen','epochs','exact','layers','heads','d_ffnn','scaled','eps','lr'])

    with open(path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([runid, args.lan, args.train_length, args.test_length, args.size, args.varlen, args.epochs, args.exact, args.layers, args.heads, args.d_ffnn, args.scaled, args.eps, args.lr])

def log_results(runid, args, train_l, val_l, train_acc, val_acc):

    from pathlib import Path
    path = Path("data/results.csv")
    if not path.exists():
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['runid', 'epoch', 'trainloss', 'vallos', 'trainacc', 'valacc'])

    for i, row in enumerate(zip(train_l, val_l, train_acc, val_acc)):
        with open(path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([runid, i+1] + list(row))

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    # training related
    ap.add_argument('--lan', type=str, dest='lan', default='first', help='Possible values: ["first", "parity", "one", "palindrome", "dyck1", "dyck2"]')
    ap.add_argument('--train_length', dest="train_length", type=int, default=10)
    ap.add_argument('--test_length', dest="test_length", type=int, default=1000)
    ap.add_argument('--size', type=int, default=100)
    ap.add_argument('--varlen', type=bool, default=False)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=10)
    # model related
    ap.add_argument('--exact', dest='exact', type=bool, default=False, help='Use the exact solution or not.')
    ap.add_argument('--layers', dest='layers', type=int, default=2)
    ap.add_argument('--heads', dest='heads', type=int, default=1)
    ap.add_argument('--d_ffnn', type=int, default=64)
    ap.add_argument('--scaled', type=bool, default=False, help='Log-length scaled attention')
    ap.add_argument('--eps', type=float, default=1e-5, help='Layer normalization value')
    ap.add_argument('--lr', type=float, default=0.0003, help='Training learning rate')

    args = ap.parse_args()
    main(args)
    