import argparse
import random
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', dest='data_type', type=str, default='first', help='Type of data to generate, either \'first\', \'parity\', \'one\', or \'palindrome\'.')
parser.add_argument('--train_size', dest='train_size', type=int, default=1000, help='The number of training examples.')
parser.add_argument('--test_size', dest='test_size', type=int, default=100, help='The number of test examples.')
parser.add_argument('--train_len', dest='train_length', type=int, default=100, help='The length of each training example. If variable_length is set, the max length of each training example.')
parser.add_argument('--test_len', dest='test_length', type=int, default=100, help='The length of each test example. If variable_length is set, the max length of each test example.')
parser.add_argument('--variable_length', dest='variable_length', type=bool, default=False, help='If set, the length of each sequence will be uniformly random between 2 and the length specified in train_len or test_len.')
args = parser.parse_args()

cols = ['sequence', 'label']
train_rows = []
test_rows = []

def get_row(max_len):
    seq_len = random.randrange(2, max_len + 1) if args.variable_length else max_len

    if args.data_type == 'first':
        sequence = [random.randrange(2) for _ in range(seq_len)]
        label = sequence[0] == 1
    elif args.data_type == 'parity':
        sequence = [random.randrange(2) for _ in range(seq_len)]
        label = sum(sequence) % 2 == 0
    elif args.data_type == 'one':
        sequence = [0 for _ in range(seq_len)]
        index = random.randrange(len(sequence))
        sequence[index] = (sequence[index] + 1) % 2
        label = True
        if(random.randrange(2) == 1):
            index = random.randrange(len(sequence))
            sequence[index] = (sequence[index] + 1) % 2
            label = False
    elif args.data_type == 'palindrome':
        prefix = [random.randrange(2) for _ in range(seq_len // 2)]
        sequence = prefix + [i for i in reversed(prefix)]
        label = True
        if(random.randrange(2) == 1):
            index = random.randrange(len(sequence))
            sequence[index] = (sequence[index] + 1) % 2
            label = False
    sequence = '$' + ''.join([str(i) for i in sequence])
    return [sequence, label]

random.seed(1)

train_rows = [get_row(args.train_length) for _ in range(args.train_size)]
test_rows = [get_row(args.test_length) for _ in range(args.test_size)]

os.makedirs(args.data_type, exist_ok=True)

variable_str = 'var' if args.variable_length else ''

with open(f'{args.data_type}/train_n{args.train_size}_l{args.train_length}{variable_str}.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(cols)
    writer.writerows(train_rows)

with open(f'{args.data_type}/test_n{args.test_size}_l{args.test_length}{variable_str}.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(cols)
    writer.writerows(train_rows)
