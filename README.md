# Overcoming Theoretical Limitations of Soft Attention

Authors: Giacomo Camposampiero, Bernhard Hilmarsson, Franz Nowak, and Clemente Pasti

Date: 31st July 2022

This README provides an overview of the code developed for our project on limitations of soft attention.

The dependencies needed to run our code are listed in ``requirements.txt``.

## Exact Experiments
To recreate our results for learned implementations run ``learned_experiments.sh``

## Exact Experiments
To recreate our results for exact implementations run ``exact_experiments.sh``

## Results
Results of experiments are stored in the ``data`` directory. Where ``data/models.csv`` contains information on the model and ``data/results.csv`` contain information per epoch, these can be linked with runid.

## Example of floating point precision error
Running ``floating_point_error.py`` will show a similar floating point error that will affect the results of Palindrome (for perfect precision, the result should be 0).

## Feature list
- [x] First Learned
- [x] First Exact 
- [x] Parity Learned
- [X] Parity Exact
- [x] ONE Learned
- [x] ONE Exact
- [x] Palindrome Learned
- [x] Palindrome Exact
