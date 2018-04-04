import argparse
from datetime import datetime
from functools import reduce

from pyspark import SparkConf, SparkContext
from utils import get_training_set, get_stop_words

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


conf = SparkConf().setAppName(f'Phase2-{datetime.now()}')
sc = SparkContext(conf=conf)
training_set = get_training_set(sc, args.training, sample=args.sample)
input_words = open(args.input, 'r').readline().lower().strip().split(' ')

def get_probability(words, place):
    print(place)
    tweets_from_place = training_set.filter(lambda x: x[0] == place)
    parts = [tweets_from_place.filter(lambda x: word in x[1]).count()/tweets_from_place.count() for word in words]
    return (tweets_from_place.count()/training_set.count()) * reduce(lambda x, y: x*y, parts)

places = training_set.filter(lambda x: any([word in input_words for word in x[1]])).map(lambda x: x[0]).distinct().collect()
probabilities = sc.parallelize([(place, get_probability(input_words, place)) for place in places]).filter(lambda x: x[1] > 0)

print(probabilities.collect())


#TODO
"""
Stopwords?

Find distinct places
Count each place
Count each word for each place

Create calculation, but how
"""