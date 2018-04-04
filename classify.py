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
training_count = training_set.count()
input_words = open(args.input, 'r').readline().lower().strip().split(' ')

places = training_set.filter(lambda x: any([word in input_words for word in x[1]])).map(lambda x: x[0]).distinct()
places_list = places.collect()
temp_set = training_set.filter(lambda x: x[0] in places_list)

print('number of places: ', places.count())

def get_probability(words, place):
    print(place)
    tweets_from_place = temp_set.filter(lambda x: x[0] == place)
    count = tweets_from_place.count()
    def counter(x, y):
        for i in range(len(words)):
            if words[i] in y:
                x[i] += 1
        return x

    parts = tweets_from_place.aggregateByKey(
        [0]*len(words),
        lambda x, y: counter(x, y),
        lambda rdd1, rdd2: (rdd1 + rdd2)
    ).map(lambda x: x[1]).collect()[0]

    return (count/training_count) * (reduce(lambda x, y: x*y, parts)/(count**len(words)))

    #parts = [tweets_from_place.filter(lambda x: word in x[1]).count()/count for word in words]
    #return (count/training_count) * reduce(lambda x, y: x*y, parts)

probabilities = sc.parallelize([(place, get_probability(input_words, place)) for place in places_list]).filter(lambda x: x[1] > 0)

output_file = open(args.output, 'w')
if probabilities.count() > 0:
    max_prob = probabilities.max()[1]
    top_places = probabilities.filter(lambda x: x[1] == max_prob).collect()
    for place in top_places:
        output_file.write(f'{place[0]}\t')
    output_file.write(str(max_prob))
else:
    output_file.write('')
output_file.close()
