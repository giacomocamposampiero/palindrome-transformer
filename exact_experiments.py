import argparse

from src.transformer import FirstExactTransformer, OneExactTransformer, ParityExactTransformer, PalindromeExactTransformer
from src.dataset import Dataset
from src.validation import Validator

import torch
import os
import csv

def main(args):

    seed = 477

    if args.lan == 'first':
        vocab = ["0", "1", "$"]
        d_model = 6
        transformer = FirstExactTransformer(len(vocab), d_model, args.normalize, args.eps)
    elif args.lan == 'parity':
        vocab = ["0", "1", "$"]
        d_model = 10
        transformer = ParityExactTransformer(len(vocab), d_model, args.normalize, args.eps)
    elif args.lan == 'one':
        vocab = ["0", "1", "$"]
        d_model = 7
        print(args.normalize)
        transformer = OneExactTransformer(len(vocab), d_model, args.normalize, args.eps)
    elif args.lan == 'palindrome':
        vocab = ["0", "1", "$", "&"]
        d_model = 12
        transformer = PalindromeExactTransformer(len(vocab), d_model, args.normalize, args.eps, args.palindrome_error)
    else:
        raise ValueError(f"{args.lan} language not supported")

        
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
    testset = Dataset(runid, args.size, args.test_length,  random_seed=seed,  train=False, data_type=args.lan, variable_lenght=False)

    validator = Validator(runid, transformer, vocab, testset, verbose=1)
    val_l, val_acc = validator.validate()

    log_results(runid, args, val_l, val_acc)

def log_model(runid, args):

    from pathlib import Path
    path = Path("data/models.csv")
    if not path.exists():
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['runid', 'lan', 'test_length', 'size', 'normalize', 'eps', 'palindrome_error'])

    with open(path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([runid, args.lan, args.test_length, args.size, args.normalize, args.eps, args.palindrome_error])

def log_results(runid, args, val_l, val_acc):

    from pathlib import Path
    path = Path("data/results.csv")
    if not path.exists():
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['runid', 'vallos', 'valacc'])

    with open(path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([runid, val_l, val_acc])

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    # training related
    ap.add_argument('--lan', type=str, dest='lan', default='first', help='Possible values: ["first", "parity", "one", "palindrome"]')
    ap.add_argument('--test_length', dest="test_length", type=int, default=1000)
    ap.add_argument('--size', type=int, default=100)

    # model related
    ap.add_argument('--normalize', type=str2bool, default=False, help='Using normalization on layers')
    ap.add_argument('--eps', type=float, default=1e-5, help='Layer normalization value')
    ap.add_argument('--palindrome_error', type=float, default=1e-7, help='Error margin for palindrome classification')

    args = ap.parse_args()
    main(args)
 
