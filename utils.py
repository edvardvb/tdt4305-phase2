from datetime import datetime
from pyspark import SparkConf, SparkContext
from constants import header

def get_training_set(context, path, sample=False):
    tweets = context.textFile(path).map(lambda x: x.split('\t'))
    tweets = tweets.sample(False, 0.1, 5) if sample else tweets
    return tweets.map(lambda x: (x[header.index('place_name')], x[header.index('tweet_text')].lower().split(' ')))

def setup(training_path, sample, input_path):
    conf = SparkConf().setAppName(f'Phase2-{datetime.now()}')
    sc = SparkContext(conf=conf)
    training_set = get_training_set(sc, training_path, sample=sample)
    training_set_count = training_set.count()
    input_words = open(input_path, 'r').readline().lower().strip().split(' ')
    input_words_count = len(input_words)
    return (training_set, training_set_count, input_words, input_words_count)