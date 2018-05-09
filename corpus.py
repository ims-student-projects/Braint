from tweet import Tweet

class Corpus(object):
    """ A datastructure to store Tweet objects.

        The files from which the data to store in Tweet/Corpus datastructure is read in
        is assumed to have the following format:

            The file containg the tweets: each line consists of two tab separated columns,
                                          in the first column the predicted label for this tweet
                                          and in the second the text of this tweet.

            The file containg the gold labels: one label on each line no empty lines in between.

            Both files have the same length and the lines in the files correspond to each other
            (e.g. The first line of the gold label file contains the label for the tweet on the first line of the other file).
    """

    def __init__(self, filename_tweets:str=None, filename_gold_labels:str=None):
        """ Inits the Corpus.

            Args:
                (optional) filename_tweets: the name of the file containing the tweets and predicted labels.
                (optional) filename_gold_labels: the name of the file containg the gold labels.
        """
        self.__corpus = []
        self.__curr = 0 # counter for iterator
        if filename_tweets and filename_gold_labels:
            self.__read_files(filename_tweets, filename_gold_labels)

    def __iter__(self):
        return iter(self.__corpus)

    def __next__(self):
        if self.__curr >= self.length():
            raise StopIteration
        else:
            self.__curr += 1
            return self.get_ith(self.__curr - 1)

    def __read_files(self, filename_tweets : str, filename_gold_labels : str):
        with open (filename_tweets, 'r') as tweet_file, \
            open (filename_gold_labels) as gold_label_file:

            #assert len(tweet_file.readlines()) == len(gold_label_file.readlines()), "File lengths do not match"

            for tweet, gold_label in zip(tweet_file, gold_label_file):
                linesplit = tweet.split('\t')
                pred_label = linesplit[0].strip()
                text = linesplit[1].strip()
                gold_label = gold_label.strip()

                tweet_obj = Tweet(text, gold_label, pred_label)
                self.__corpus.append(tweet_obj)

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
