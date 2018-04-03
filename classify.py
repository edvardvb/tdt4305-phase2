import argparse
from datetime import datetime

from pyspark import SparkConf, SparkContext

from constants import header
from utils import get_training_set, get_input_set, get_stop_words

parser = argparse.ArgumentParser()
parser.add_argument('-training', '-t', help='Path to training set', type=str)
parser.add_argument('-input', '-i', help='Path to input file', type=str)
parser.add_argument('-output', '-o', help='Path to output file', type=str)
parser.add_argument('-sample', '-s', help='Used to sample the training set for shorter computation time', action='store_true')
args = parser.parse_args()

print(args.training)
print(args.input)
print(args.output)
print(args.sample)

def setup():
    conf = SparkConf().setAppName(f'Phase2-{datetime.now()}')
    sc = SparkContext(conf=conf)
    training_set = get_training_set(sc, args.training, sample=args.sample)
    input_text = open(args.input, 'r').readline().strip()
    return (conf, sc, training_set, input_text)

#TODO
"""
Stopwords?

Find distinct places
Count each place
Count each word for each place

Create calculation, but how
"""