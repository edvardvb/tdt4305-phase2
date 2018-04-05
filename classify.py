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
parser.add_argument('-pretty', '-p', help='Prints extra information while executing if set', action='store_true')
args = parser.parse_args()

if args.pretty:
    print('🚀  Training set:', args.training)
    print('🚀  Input file:', args.input)
    print('🚀  Output file:', args.output)
    print('🚀  Sampling:', args.sample)

conf = SparkConf().setAppName(f'Phase2-{datetime.now()}')
sc = SparkContext(conf=conf)
training_set = get_training_set(sc, args.training, sample=args.sample)
training_count = training_set.count()
input_words = open(args.input, 'r').readline().lower().strip().split(' ')
input_words_count = len(input_words)

not_places = training_set\
    .aggregateByKey([], lambda x, y: x + y, lambda x, y: x + y)\
    .filter(lambda x: any([word not in x[1] for word in input_words]))\
    .map(lambda x: x[0]).collect()

if args.pretty: print('🔍  Number of places that don\'t have any relevant tweets:', len(not_places))


places = training_set\
    .filter(lambda x: any([word in input_words for word in x[1]]))\
    .map(lambda x: x[0])\
    .distinct()\
    .filter(lambda x: x not in not_places)
places_list = places.collect()

temp_set = training_set.filter(lambda x: x[0] in places_list)

if args.pretty: print('💁  Number of places with relevant tweets:', places.count())


def counter(x, y):
    for i in range(input_words_count):
        if input_words[i] in y:
            x[i] += 1
    return x

def get_probability(i, place):
    if args.pretty: print('==============')
    if args.pretty: print('🗺 ', i, place)
    tweets_from_place = temp_set.filter(lambda x: x[0] == place).map(lambda x: x[1])
    count = tweets_from_place.count()
    if args.pretty: print('📚  Number of tweets:', count)
    parts = tweets_from_place.aggregate(
        [0]*input_words_count,
        lambda x, y: counter(x, y),
        lambda rdd1, rdd2: [rdd1[i] + rdd2[i] for i, j in enumerate(rdd1)]
    )
    if args.pretty: print('📊  Word counts:', parts)
    probability = (count/training_count) * (reduce(lambda x, y: x*y, parts)/(count**input_words_count))
    if args.pretty: print ('🎲  Probability:', probability)
    return probability


probabilities = sc.parallelize([(place, get_probability(i, place)) for i, place in enumerate(places_list)]).filter(lambda x: x[1] > 0)

output_file = open(args.output, 'w')
if probabilities.count() > 0:
    max_prob = probabilities.map(lambda x: x[1]).max()
    top_places = probabilities.filter(lambda x: x[1] == max_prob).collect()
    for place in top_places:
        output_file.write(f'{place[0]}\t')
    output_file.write(str(max_prob))
else:
    output_file.write('')
output_file.close()
