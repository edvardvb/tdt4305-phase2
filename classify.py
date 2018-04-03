import argparse
from datetime import datetime

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
input_text = open(args.input, 'r').readline().strip()

places = training_set.map(lambda x: x[0]).distinct().take(5)

place_counts = training_set.keyBy(lambda x: x[0]).sortByKey().countByKey()
# dette burde gjøres med en aggregering i stedet, så får man en RDD tilbake istedet for en dict.
# Da slipper man den over for å finne distinkte også

print(places)
print(place_counts)


#TODO
"""
Stopwords?

Find distinct places
Count each place
Count each word for each place

Create calculation, but how
"""