#!/usr/bin/env bash

# FIRST learning experiments
python data_generator.py --data_type "first" --train_size 10000 --test_size 2000 --train_len 10 --test_len 1000                       
python data_generator.py --data_type "first" --train_size 10000 --train_len 30
python data_generator.py --data_type "first" --train_size 10000 --train_len 100
python data_generator.py --data_type "first" --train_size 10000 --train_len 300

# PARITY learning experiments
python data_generator.py --data_type "parity" --train_size 10000 --test_size 2000 --train_len 10 --test_len 1000                       
python data_generator.py --data_type "parity" --train_size 10000 --train_len 30
python data_generator.py --data_type "parity" --train_size 10000 --train_len 100
python data_generator.py --data_type "parity" --train_size 10000 --train_len 300

# ONE learning experiments
python data_generator.py --data_type "one" --train_size 10000 --test_size 2000 --train_len 10 --test_len 1000                       
python data_generator.py --data_type "one" --train_size 10000 --train_len 30
python data_generator.py --data_type "one" --train_size 10000 --train_len 100
python data_generator.py --data_type "one" --train_size 10000 --train_len 300

# PALINDROME learning experiments
python data_generator.py --data_type "palindrome" --train_size 10000 --test_size 2000 --train_len 10 --test_len 1000                       
python data_generator.py --data_type "palindrome" --train_size 10000 --train_len 30
python data_generator.py --data_type "palindrome" --train_size 10000 --train_len 100
python data_generator.py --data_type "palindrome" --train_size 10000 --train_len 300
