import numpy
import argparse
import random
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', dest='data_type', type=str, default='first', help='Type of data to generate, either \'first\', \'parity\', \'one\', \'palindrome\' or \'dyck1\'.')
parser.add_argument('--train_size', dest='train_size', type=int, default=1000, help='The number of training examples.')
parser.add_argument('--test_size', dest='test_size', type=int, default=100, help='The number of test examples.')
parser.add_argument('--train_len', dest='train_length', type=int, default=100, help='The length of each training example. If variable_length is set, the max length of each training example.')
parser.add_argument('--test_len', dest='test_length', type=int, default=100, help='The length of each test example. If variable_length is set, the max length of each test example.')
parser.add_argument('--variable_length', dest='variable_length', type=bool, default=False, help='If set, the length of each sequence will be uniformly random between 2 and the length specified in train_len or test_len.')
args = parser.parse_args()

cols = ['sequence', 'label']
train_rows = []
test_rows = []

# returns all indecs of element a in list
def get_indecies(ls, a):
    indecies = []
    for x in range(len(ls)):
        if ls[x] == a:
            indecies.append(x)
    return indecies

# Length is how many matching brackets should appear (), would be 1
# alphabet is array of tuple of matching pairs i.e. [ ( '(', ')' ), ('{', '}') ]
def cfg_generate(length, alphabet= [ ( '(', ')' ), ('{', '}') ]):
    if length == 0:
        return []

    if random.randint(0,1) == 1:
        # case (S)
        selected = random.randrange(len(alphabet))
        return [*alphabet[selected][0], *cfg_generate(length-1, alphabet), *alphabet[selected][1] ]
    else:
        # case SS
        left_length = random.randrange(length+1)
        right_length = length - left_length
        return [*cfg_generate(left_length, alphabet), *cfg_generate(right_length, alphabet)]

# validates any sequence including {}, () and 0,1
def valid_bracket_sequence(seq):
    match = {}
    match['{'] = '}'
    match['('] = ')'
    match['0'] = '1'

    stack = []
    for x in seq:
        if x in match:
            stack.append(x)
        else:
            if len(stack) == 0:
                return False
            else:
                last = stack.pop()
                if match[last] != x:
                    return False

    if len(stack) == 0:
        return True
    return False




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
        number_of_ones = min(seq_len, numpy.random.poisson(lam=1.5, size=1)[0])
        indecies = [x for x in range(seq_len)]
        selected_indecies = random.sample(indecies, number_of_ones)
        for index in selected_indecies:
            sequence[index] = 1
        label = True
        if number_of_ones != 1:
            label = False
    elif args.data_type == 'palindrome':
        prefix = [random.randrange(2) for _ in range(seq_len // 2)]
        sequence = prefix + [i for i in reversed(prefix)]
        label = True
        if(random.randrange(2) == 1):
            index = random.randrange(len(sequence))
            sequence[index] = (sequence[index] + 1) % 2
            label = False
    elif args.data_type == 'dyck1':
        half_len = seq_len // 2
        sequence = cfg_generate(half_len, [('(', ')')])
        label = True
        # Validation of bracket sequence (can be removed)
        assert valid_bracket_sequence(sequence)

        if random.randrange(2) == 1:
            label = False
            # Flip randomly, both types i.e ( then ), until bracket sequence is no longer valid
            while valid_bracket_sequence(sequence):
                # flip (
                indecies = get_indecies(sequence, '(')
                index = indecies[random.randrange(len(indecies))]
                sequence[index] = ')'

                # flip )
                indecies = get_indecies(sequence, ')')
                index = indecies[random.randrange(len(indecies))]
                sequence[index] = '('

    elif args.data_type == 'dyck2':
        half_len = seq_len // 2
        sequence = cfg_generate(half_len)
        label = True

        assert valid_bracket_sequence(sequence)
        if random.randrange(2) == 1:
            label = False
            while valid_bracket_sequence(sequence):
                if '(' in sequence:
                    # flip (
                    indecies = get_indecies(sequence, '(')
                    index = indecies[random.randrange(len(indecies))]
                    sequence[index] = ')'

                    # flip )
                    indecies = get_indecies(sequence, ')')
                    index = indecies[random.randrange(len(indecies))]
                    sequence[index] = '('
                if '{' in sequence:
                     # flip {
                    indecies = get_indecies(sequence, '{')
                    index = indecies[random.randrange(len(indecies))]
                    sequence[index] = '}'

                    # flip )
                    indecies = get_indecies(sequence, '}')
                    index = indecies[random.randrange(len(indecies))]
                    sequence[index] = '{'



    sequence = '$' + ''.join([str(i) for i in sequence])
    return [sequence, label]

random.seed(1)
numpy.random.seed(1)

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
