

for test_length in 10 20 40 80 160 320 640 1000 2000 4000 10000
do
	python3 exact_experiments.py --lan first --test_length $test_length --size 500 
done

for test_length in 10 20 40 80 160 320 640 1000 2000 4000 10000
do
	python3 exact_experiments.py --lan parity --test_length $test_length --size 500 
done

for test_length in 10 20 40 80 160 320 640 1000 2000 4000 10000
do
	python3 exact_experiments.py --lan one --test_length $test_length --size 500 
done

for test_length in 2 4 8 16 32 40 50 60 70 80 90 100
do
	python3 exact_experiments.py --lan palindrome --test_length $test_length --size 500 
done

for test_length in 1 3 5 7 15 31 39 49 59 69 79 89 99
do
	python3 exact_experiments.py --lan palindrome --test_length $test_length --size 500 
done
