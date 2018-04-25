from itertools import zip_longest

import tweet

class Corpus(object):

    def __init__(self, filename_tweets=None, filename_gold_labels=None):
        self.__corpus = []
        self.__curr = 0 # counter for iterator
        if filename_tweets and filename_gold_labels:
            self.read_files(filename_tweets, filename_gold_labels)

    def __iter__(self):
        return iter(self.__corpus)

    def __next__(self):
        if self.__curr >= self.length():
            raise StopIteration
        else:
            self.__curr += 1
            return self.get_ith(self.__curr - 1)

    def read_files(self, filename_tweets, filename_gold_labels):
        with open (filename_tweets, 'r') as tweet_file, \
            open (filename_gold_labels) as gold_label_file:
            for tweet, gold_label in zip_longest(tweet_file, gold_label_file):
                linesplit = tweet.split('\t')
                pred_label = linesplit[0].strip()
                text = linesplit[1].strip()
                gold_label = gold_label.strip()

                tweet_obj = Tweet.Tweet(text, gold_label, pred_label)
                self.append(tweet_obj)

    def length(self):
        return len(self.__corpus)

    def get_ith(self, i):
        return self.__corpus[i]

    # @override
    def append(self, tweet_obj, pos=None):
        if pos is not None:
            self.__corpus.insert(pos, tweet_obj)
        else:
            self.__corpus.append(tweet_obj)
