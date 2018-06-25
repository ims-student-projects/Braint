from tweet import Tweet
from random import shuffle

class Corpus(object):
    """ A datastructure to store Tweet objects.

        The files from which the data to store in Tweet/Corpus datastructure is
        read is assumed to have the following format:

            (Mandatory) The file containg the tweets and predicted labels: each
                line consists of two tab separated columns, in the first column
                the predicted label for this tweet and in the second the text of
                this tweet.

            (Optional) The file containg the gold labels: one label on each line
                no empty lines in between.

            Both files are assumed to have the same length and the lines in the
            files correspond to each other (e.g. The first line of the gold
            label file contains the label for the tweet on the first line of the
            other file).
    """

    def __init__(self, filename_tweets:str, filename_gold_labels:str=None, \
                print_distr:bool=False):
        """ Inits the Corpus.

            Args:
                filename_tweets: the name of the file containing the tweets and
                    predicted labels.
                (optional) filename_gold_labels: the name of the file containg
                    the gold labels.
                (optional) print statistics about class distribution.
        """
        self.__corpus = []
        self.__curr = 0 # counter for iterator
        self.__distr = {} # stats about class distribution in corpus
        if filename_tweets and filename_gold_labels:
            self.__read_test_files(filename_tweets, filename_gold_labels)
        elif filename_tweets:
            self.__read_train_file(filename_tweets)
        if print_distr:
            self.print_distr()
        self.__all_feature_names = []


    def __iter__(self):
        return iter(self.__corpus)


    def __next__(self):
        if self.__curr >= self.length():
            raise StopIteration
        else:
            self.__curr += 1
            return self.get_ith(self.__curr - 1)


    def __read_train_file(self, filename_tweets:str):
        with open (filename_tweets, 'r') as train_file:
            for line in train_file:
                line = line.split('\t')
                gold_label = line[0].strip()
                self.__distr[gold_label] = self.__distr.get(gold_label, 0) + 1
                text = line[1].strip()
                tweet_obj = Tweet(text, gold_label, None)
                self.__corpus.append(tweet_obj)


    def __read_test_files(self, filename_tweets:str, filename_gold_labels:str):
        with open (filename_tweets, 'r') as tweet_file, \
            open (filename_gold_labels) as gold_label_file:
            for tweet, gold_label in zip(tweet_file, gold_label_file):
                linesplit = tweet.split('\t')
                pred_label = linesplit[0].strip()
                text = linesplit[1].strip()
                gold_label = gold_label.strip()
                self.__distr[gold_label] = self.__distr.get(gold_label, 0) + 1
                tweet_obj = Tweet(text, gold_label, pred_label)
                self.__corpus.append(tweet_obj)


    def get_distr(self):
        return self.__distr


    def print_distr(self):
        self.__distr['total'] = sum(self.__distr.values())
        emotions = list(self.__distr.keys())
        print('{}'.format('\t\t'.join(emotions)))
        all_data = []
        for e in emotions:
            perc = round((self.__distr[e]*100 / self.__distr['total']),1) if \
                self.__distr['total'] != 0 else 0
            data = '{}% ({})'.format(perc, self.__distr[e])
            all_data.append(data)
        print('{}\n'.format('\t'.join(all_data)))


    def set_all_feature_names(self, feature_names):
        self.__all_feature_names = feature_names


    def get_all_feature_names(self):
        return self.__all_feature_names


    def length(self):
        """ Gets the number of tweets in the corpus.

            Returns:
                The number of tweets storde in this corpus.
        """
        return len(self.__corpus)


    def get_ith(self, i : int):
        """ Gets the ith tweet in this corpus.

            Return:
                A tweet object.
        """
        return self.__corpus[i]


    def shuffle(self):
        shuffle(self.__corpus)
