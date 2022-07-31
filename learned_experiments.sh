 # FIRST generalization experiment
python learned.py --lan first --train_length 10 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled False
python learned.py --lan first --train_length 300 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled False
python learned.py --lan first --train_length 30 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled False
python learned.py --lan first --train_length 100 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled False

python learned.py --lan first --train_length 10 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled True
python learned.py --lan first --train_length 300 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled True
python learned.py --lan first --train_length 30 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled True
python learned.py --lan first --train_length 100 --test_length 1000 --size 100 --runs 20 --epochs 1000 --scaled True

python learned.py --lan one --train_length 10 --test_length 1000 --size 100 --runs 1 --epochs 1000 --scaled False --varlen True
python learned.py --lan one --train_length 300 --test_length 1000 --size 100 --runs 1 --epochs 1000 --scaled False --varlen True
python learned.py --lan one --train_length 30 --test_length 1000 --size 100 --runs 1 --epochs 1000 --scaled False --varlen True
python learned.py --lan one --train_length 100 --test_length 1000 --size 100 --runs 1 --epochs 1000 --scaled False --varlen True


# FIRST learning experiment
python learned.py --lan first --train_length 10 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 20 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 30 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 50 --test_length 50 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 70 --test_length 70 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 100 --test_length 100 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 200 --test_length 200 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 300 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 400v --test_length 300 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 500 --test_length 500 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 750 --test_length 750 --size 100 --runs 10 --epochs 100 --scaled True
python learned.py --lan first --train_length 1000 --test_length 1000 --size 100 --runs 10 --epochs 100 --scaled True

# PARITY learning experiment
python learned.py --lan parity --train_length 10 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 20 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 30 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 50 --test_length 50 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 70 --test_length 70 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 100 --test_length 100 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 200 --test_length 200 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 300 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 400 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 500 --test_length 500 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 750 --test_length 750 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan parity --train_length 1000 --test_length 1000 --size 100 --runs 10 --epochs 100 --scaled False

# PALINDROME learning experiment 
python learned.py --lan palindrome --train_length 10 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 20 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 30 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 50 --test_length 50 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 70 --test_length 70 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 100 --test_length 100 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 200 --test_length 200 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 300 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 400 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 500 --test_length 500 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 750 --test_length 750 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan palindrome --train_length 1000 --test_length 1000 --size 100 --runs 10 --epochs 100 --scaled False

# ONE learning experiment 
python learned.py --lan one --train_length 10 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 20 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 30 --test_length 10 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 50 --test_length 50 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 70 --test_length 70 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 100 --test_length 100 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 200 --test_length 200 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 300 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 400 --test_length 300 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 500 --test_length 500 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 750 --test_length 750 --size 100 --runs 10 --epochs 100 --scaled False
python learned.py --lan one --train_length 1000 --test_length 1000 --size 100 --runs 10 --epochs 100 --scaled False
