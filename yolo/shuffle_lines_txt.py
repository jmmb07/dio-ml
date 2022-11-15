import random

lines = open('output.txt').readlines()
random.shuffle(lines)
open('train.txt', 'w').writelines(lines)