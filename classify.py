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

training_set, training_set_count, input_words, input_words_count, sc = setup(args.training, args.sample, args.input)

def counter(x, y):
    """ Count the number of occurrences of each word in input tweet
    Args: 
        x (list): A list the same size as number of words in input tweet, representing the word count
        y (list): List of words in tweet from location

    Returns:
        List where each element is the count of occurrences of word from input tweet
    """
    
    for i in range(input_words_count):
        if input_words[i] in y:
            x[i] += 1
    return x

# Places is a RDD where values for the location keys are represented as tuples, containing a list of wordcounts of each word 
# in the input tweet for that location, together with the location’s total number of tweets. 
places = training_set\
    .aggregateByKey(
        ([0]*input_words_count, 0), # Format of the initial value for the keys
        lambda x, y: (counter(x[0], y), x[1] +1), # Combine the types of values in the RDD into desired resulting type: a tuple
        lambda rdd1, rdd2: (
                # Combines the list of the first with the list of the second by summing the respective values of each list and combines the count of tweets by simply adding them together
                [rdd1[0][i] + rdd2[0][i] for i, j in enumerate(rdd1[0])],
                rdd1[1] + rdd2[1]
        )
    # Filter with respect to places where all words from the input tweet occur at least once   
    )\
    .filter(lambda x: all(x[1][0]))\
    .sortByKey()\
    .cache() 

places_list = places.map(lambda x: x[0]).collect() # Create RDD with location names

if args.pretty: print('💁  Number of places with relevant tweets:', len(places_list))

def get_probability(i, place):
    """ Calculate probability for input tweet originating in a given locaction
    Args: 
        i (int): Location number i found relevant to the tweet
        place (str): Name of location

    Returns:
        The calculated probability
    """

    if args.pretty: print('==============')
    if args.pretty: print('🗺 ', i, place)
    word_counts, no_of_tweets_from_place = places.lookup(place)[0]

    if args.pretty: print('📚  Number of tweets:', no_of_tweets_from_place)
    if args.pretty: print('📊  Word counts:', word_counts)
    
    probability = (no_of_tweets_from_place/training_set_count) * (reduce(lambda x, y: x*y, word_counts)/(no_of_tweets_from_place**input_words_count)) # Use Naive Bayes formula
    if args.pretty: print ('🎲  Probability:', probability)

    return probability

# Calculate probabilities for all places in places_list
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
