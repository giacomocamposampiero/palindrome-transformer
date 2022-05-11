import argparse
import random
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', dest='data_type', type=str, default="first", help='Type of data to generate, either \'first\', \'parity\', or \'palindrome\'')
parser.add_argument('--train_len', dest='train_length', type=int, default=100)
parser.add_argument('--test_len', dest='test_length', type=int, default=100)
parser.add_argument('--train_size', dest='train_size', type=int, default=1000)
parser.add_argument('--test_size', dest='test_size', type=int, default=100)
args = parser.parse_args()

cols = ["sequence", "label"]
train_rows = []
test_rows = []

def get_row(seq_len):
    if args.data_type == "first":
        sequence = [random.randrange(2) for i in range(seq_len)]
        label = sequence[0] == 1
    elif args.data_type == "parity":
        sequence = [random.randrange(2) for i in range(seq_len)]
        label = sum(sequence) % 2 == 0
    elif args.data_type == "palindrome":
        prefix = [random.randrange(2) for i in range(seq_len)]
        sequence = prefix + [i for i in reversed(prefix)]
        label = True
        if(random.randrange(2) == 1):
            index = random.randrange(len(sequence))
            sequence[index] = (sequence[index] + 1 ) % 2
            label = False
    sequence = '$' + ''.join([str(i) for i in sequence])
    return [sequence, label]


train_rows = [get_row(args.train_length) for _ in range(args.train_size)]
test_rows = [get_row(args.test_length) for _ in range(args.test_size)]

os.makedirs("data/" + args.data_type, exist_ok=True)

with open("data/" + args.data_type + '/train.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(cols)
    writer.writerows(train_rows)

with open("data/" + args.data_type + '/test.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(cols)
    writer.writerows(train_rows)