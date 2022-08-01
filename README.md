# Overcoming Theoretical Limitations of Soft Attention

Authors: Giacomo Camposampiero, Bernhard Hilmarsson, Franz Nowak, and Clemente Pasti

Date: 31st July 2022

## Introduction

In response to the limitations postulated by [Hahn (2020)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00306/43545/Theoretical-Limitations-of-Self-Attention-in), [Chiang and Cholak (2022)](https://aclanthology.org/2022.acl-long.527/) showed that there exist transformers that can recognize PARITY with perfect accuracy. 

In this work, we have verified their results experimentally, finding that the exact solutions do indeed work as described, but that PARITY cannot be learned. 
We then extended their results by deriving custom transformer weights that, at least in in theory, can recognize instances of the regular language ONE and the context free language PALINDROME for arbitrary input sizes n. 

In practice, however, the solution for PALINDROME does not generalise to longer sequences due to floating point precision errors.

This work was developed as course project for the course [Advanced Formal Language Theory, Spring 2022](https://rycolab.io/classes/aflt-s22/
) at ETH Zürich.

This README provides an overview of the code developed for our project on limitations of soft attention.

## Getting started with the code
To install all the required dependencies, use 
```
pip install -r requirements.txt
```

## Replicating the experiments
To replicate our results for learned implementations, use the following commands. The replication of learned experiments might take few days, depending on the machine where the code is run.
```
chmod 700 experiments/learned_experiments.sh
./experiments/learned_experiments.sh
```

To replicate our results for exact implementations, use
```
chmod 700 experiments/exact_experiments.sh
./experiments/exact_experiments.sh
```

To replicate the graphs included in our paper, use the ``plot.py`` script provided inside the directories of each experiment category.
For example, to replicate the graphs for FIRST learning experiment, use
```
python experiments/learn/plot.py
```

## Experiment and result logging
Results of experiments are stored in the ``data`` directory. 
- ``data/models.csv`` contains information on the run and the model.
- ``data/results.csv`` contain training information (validation and training loss/accuracy) for each epoch of different runs

Every traning (or testing in exact implementations) run is logged with a unique run identification number. The results and models tables can be joined on the attribute ``runid``. 

## Example of floating point precision error
Running ``experiments/floating_point_error.py`` will show a similar floating point error that affect the results of Palindrome (for perfect precision, the result should be 0).

## Feature list
- [x] First Learned
- [x] First Exact 
- [x] Parity Learned
- [X] Parity Exact
- [x] ONE Learned
- [x] ONE Exact
- [x] Palindrome Learned
- [x] Palindrome Exact
