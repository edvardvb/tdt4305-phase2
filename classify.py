import argparse
from functools import reduce

from utils import setup

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

training_set, training_set_count, input_words, input_words_count = setup(args.training, args.sample, args.input)

def counter(x, y):
    for i in range(input_words_count):
        if input_words[i] in y:
            x[i] += 1
    return x


places = training_set\
    .aggregateByKey(
        ([0]*input_words_count, 0),
        lambda x, y: (counter(x[0], y), x[1] +1),
        lambda rdd1, rdd2: (
                [rdd1[0][i] + rdd2[0][i] for i, j in enumerate(rdd1[0])],
                rdd1[1] + rdd2[1]
        )
    )\
    .filter(lambda x: all(x[1][0]))\
    .sortByKey()\
    .cache()

places_list = places.map(lambda x: x[0]).collect()
if args.pretty: print('💁  Number of places with relevant tweets:', len(places_list))

def get_probability(i, place):
    if args.pretty: print('==============')
    if args.pretty: print('🗺 ', i, place)

    no_of_tweets_from_place = places.lookup(place)[0][1]
    word_counts = places.lookup(place)[0][0]
    if args.pretty: print('📚  Number of tweets:', no_of_tweets_from_place)
    if args.pretty: print('📊  Word counts:', word_counts)

    probability = (no_of_tweets_from_place/training_set_count) * (reduce(lambda x, y: x*y, word_counts)/(no_of_tweets_from_place**input_words_count))
    if args.pretty: print ('🎲  Probability:', probability)

    return probability


probabilities = sc.parallelize([(place, get_probability(i, place)) for i, place in enumerate(places_list)])

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
