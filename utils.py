from constants import header

def get_training_set(context, path, sample=False):
    tweets = context.textFile(path).map(lambda x: x.split('\t'))
    tweets = tweets.sample(False, 0.1, 5) if sample else tweets
    return tweets.map(lambda x: (x[header.index('place_name')], x[header.index('tweet_text')].lower().split(' ')))

def get_stop_words():
    stripped = []
    with open('data/stop_words.txt', 'r') as file:
        for line in file:
            stripped.append(line.strip('\n'))
        return stripped