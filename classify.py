import argparse
from datetime import datetime

from pyspark import SparkConf, SparkContext

from constants import header
from utils import get_training_set, get_input_set, get_stop_words

parser = argparse.ArgumentParser()
parser.add_argument('-training', help='path to training set', type=str)
parser.add_argument('-input', help='path to input file', type=str)
parser.add_argument('-output', help='path to output file', type=str)
parser.add_argument('-sample', type=bool)
args = parser.parse_args()

print(args.training)
print(args.input)
print(args.output)

def setup():
    conf = SparkConf().setAppName(f'Phase2-{datetime.now()}')
    sc = SparkContext(conf=conf)
    training_set = get_training_set(sc, args.training, sample=args.sample).map(lambda x: (x[header.index('place_name')], x[header.index('tweet_text')]))
    input_set = get_input_set(sc, args.input)
    return (conf, sc, training_set, input_set)