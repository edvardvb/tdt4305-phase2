def get_training_set(context, path, sample=False):
    tweets = context.textFile(path).map(lambda x: x.split('\t'))
    return tweets.sample(False, 0.1, 5) if sample else tweets

def get_input_set(context, path):
    return context.textFile(path)

def get_stop_words():
    stripped = []
    with open('data/stop_words.txt', 'r') as file:
        for line in file:
            stripped.append(line.strip('\n'))
        return stripped