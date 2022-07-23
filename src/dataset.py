import numpy
import random
import csv
import os

class Dataset:
    """
    Dataset
    -------

    Dataset wrapper class, instance of an Iterator design pattern. 
    Samples are generated on the fly, but random seeds are used to allow replicability.
    Every sample is logged on the fly in a `csv` file in the data folder.

    Paramers
    --------

    uid : int
          unique identification number that identifies the run; MUST be the same for 
          for the dataset and the trainer

    size : int
           size of the dataset

    length : int
             length of the generated strings

    random_seed : int 
                  random seed to regulate class' non-deterministic behaviour and 
                  allow experiments replicability

    train : bool
            boolean flag to indicate whether this dataset is used for training or test

    data_type : str
                language from which dataset entries are sampled; supported languages:
                    - first
                    - parity
                    - one
                    - palindrome
                    - dyck1
                    - dyck2
    
    variable_lenght : bool
                      allow data samples to vary in size or not; if set to false, all
                      generated data samples will have same size (equal to self.size)
    """
    def __init__(self, 
                 uid : int, 
                 size : int, 
                 length : int, 
                 random_seed : int, 
                 train : bool, 
                 data_type : str = 'first', 
                 variable_lenght : bool = False
                ) -> None:

        random.seed(random_seed)
        numpy.random.seed(random_seed)
        self.uid = uid
        self.size = size
        self.length = length
        self.data_type = data_type
        self.variable_length = variable_lenght
        self.index = 0
        self.train = train
        self.epoch = 0


    def __iter__(self):
        """
        Return itself.
        """
        return self


    def __next__(self):
        """
        Iteration step in the dataset.
        """
        row = self.__get_row(self.length)
        self.index += 1
        if self.index <= self.size: 
            self.__log_row(row)
            return row[0], row[1]
        else: raise StopIteration


    def reset_index(self):
        """
        Reset the index for the iterator. Allows multiple epochs training.
        """
        self.index = 0
        self.epoch +=1


    # TODO
    def __log_row(self, row):
        """ 
        Log rows generated during training.

        Parameters
        ----------
        row : tuple[str, bool]
              tuple containing a training or validation entry; each entry has a 
              value (string) and a label (bool)
        """

        cols = ['epoch', 'sequence', 'label']

        os.makedirs("data/" + self.data_type, exist_ok=True)
        variable_str = 'var' if self.variable_length else ''

        if self.train:
            logpath = "data/" + self.data_type + f"/run{self.uid:04d}_train.csv"
        else:
            logpath = "data/" + self.data_type + f"/run{self.uid:04d}_test.csv"

        from pathlib import Path
        path = Path(logpath)

        if path.exists():
            # this is not the first row
            with open(path, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([self.epoch]+row)
        else:
            # this is the first row
            with open(path, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(cols)
                writer.writerow([self.epoch]+row)


    def __get_indices(self, ls, a):
        """
        Returns all indices of element a in list.
        """
        indecies = []
        for x in range(len(ls)):
            if ls[x] == a:
                indecies.append(x)
        return indecies

    
    def __generate_bracket_sequence(self, length, alphabet= [ ( '(', ')' ), ('{', '}') ]):
        """
        Length is how many matching brackets should appear (), would be 1
        alphabet is array of tuple of matching pairs i.e. [ ( '(', ')' ), ('{', '}') ]
        """
        if length == 0:
            return []

        if random.randint(0,1) == 1:
            # case (S)
            selected = random.randrange(len(alphabet))
            return [*alphabet[selected][0], *self.__cfg_generate(length-1, alphabet), *alphabet[selected][1] ]
        else:
            # case SS
            left_length = random.randrange(length+1)
            right_length = length - left_length
            return [self.__cfg_generate(left_length, alphabet), *self.__cfg_generate(right_length, alphabet)]

    
    def __valid_bracket_sequence(self, seq):
        """
        validates any sequence including {}, () and 0,1.
        """
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

        if len(stack) == 0: return True
        return False


    def __get_row(self, max_len):
        """
        Return pair (sample, label).
        """        
        seq_len = random.randrange(2, max_len + 1) if self.variable_length else max_len

        if self.data_type == 'first':
            sequence = [random.randrange(2) for _ in range(seq_len)]
            label = sequence[0] == 1
        elif self.data_type == 'parity':
            sequence = [random.randrange(2) for _ in range(seq_len)]
            label = sum(sequence) % 2 == 0
        elif self.data_type == 'one':
            sequence = [0 for _ in range(seq_len)]
            number_of_ones = min(seq_len, numpy.random.poisson(lam=1.5, size=1)[0])
            indecies = [x for x in range(seq_len)]
            selected_indecies = random.sample(indecies, number_of_ones)
            for index in selected_indecies:
                sequence[index] = 1
            label = True
            if number_of_ones != 1:
                label = False
        elif self.data_type == 'palindrome':
            prefix = [random.randrange(2) for _ in range(seq_len // 2)]
            sequence = prefix + [i for i in reversed(prefix)]
            label = True
            if(random.randrange(2) == 1):
                index = random.randrange(len(sequence))
                sequence[index] = (sequence[index] + 1) % 2
                label = False
        elif self.data_type == 'dyck1':
            half_len = seq_len // 2
            sequence = self.__cfg_generate(half_len, [('(', ')')])
            label = True
            # Validation of bracket sequence (can be removed)
            assert self.__valid_bracket_sequence(sequence)

            if random.randrange(2) == 1:
                label = False
                # Flip randomly, both types i.e ( then ), until bracket sequence is no longer valid
                while self.__valid_bracket_sequence(sequence):
                    # flip (
                    indecies = self.__get_indices(sequence, '(')
                    index = indecies[random.randrange(len(indecies))]
                    sequence[index] = ')'

                    # flip )
                    indecies = self.__get_indices(sequence, ')')
                    index = indecies[random.randrange(len(indecies))]
                    sequence[index] = '('

        elif self.data_type == 'dyck2':
            half_len = seq_len // 2
            sequence = self.__cfg_generate(half_len)
            label = True

            assert self.__valid_bracket_sequence(sequence)
            if random.randrange(2) == 1:
                label = False
                while self.__valid_bracket_sequence(sequence):
                    if '(' in sequence:
                        # flip (
                        indecies = self.__get_indices(sequence, '(')
                        index = indecies[random.randrange(len(indecies))]
                        sequence[index] = ')'

                        # flip )
                        indecies = self.__get_indices(sequence, ')')
                        index = indecies[random.randrange(len(indecies))]
                        sequence[index] = '('
                    if '{' in sequence:
                        # flip {
                        indecies = self.__get_indices(sequence, '{')
                        index = indecies[random.randrange(len(indecies))]
                        sequence[index] = '}'

                        # flip )
                        indecies = self.__get_indices(sequence, '}')
                        index = indecies[random.randrange(len(indecies))]
                        sequence[index] = '{'

        sequence = '$' + ''.join([str(i) for i in sequence])
        return [sequence, label]


# # DEBUG DATASET CLASS
# dataset =  Dataset(0, 20, 2,  42, True, 'first', False)
# for row in dataset:
#     print(row)
# dataset.reset_index()
# print("+"*50)
# for row in dataset:
#     print(row)