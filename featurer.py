from tokenizer import Tokenizer
from math import log10

class Featurer():
    """
    Extracts feature vectors for tweets in a corpus. Features are terms in
    tweets represented by their tf-idf score. An additional feature named
    "THETA" is added with a constant value of 1 (necessary for the perceptron).

    Expected input is an object of class Corpus. This class also uses the class
    Tokenizer to convert tweet text into tokens/terms.

    Results are fed back into the corpus and tweets. To access them use:
    corpus.get_all_features() -- for a list of all feature labels
    tweet.get_features() -- to get the features of each individual tweet

    Featurer can be used as a generator to iterate over tweet features.
    A list of all features (i.e. terms in corpus) can be accessed by the
    method get_all_features().
    """

    def __init__(self, corpus):
        self.__corpus = corpus
        self.__size = 0  # corpus size, i.e. number of tweets
        self.__term_idfs = {} # dict with term-idf-score pairs
        self.get_idf_scores()  # calculate idfs for each term in corpus


    def __iter__(self):
        """
        Make object iterable
        """
        self.__curr = 0
        self.__tweets = iter(self.corpus)
        return self


    def __next__(self):
        """
        Return a tuple of label (emotion), tweet text and features (tf-dfs)
        """
        if self.__curr == self.__size:
            raise StopIteration
        else:
            self.__curr += 1
            tweet = next(self.__tweets)
            label = tweet.get_gold_label()
            features = self.get_tf_idf(tweet)
            text = tweet.get_text()
            return (label, text, features)


    def get_idf_scores(self):
        """
        Get a list of all terms in corpus and calculate for each term a df
        score (number of tweets in which the term occurs). Then convert df to
        idf.
        """

        #  Extract terms from corpus and calculate document frequency for each.
        #  Simulataneously collection size is calculated.
        for tweet in self.__corpus:
            self.__size += 1
            terms = Tokenizer(tweet.get_text()).get_terms()  # TODO: adapt to Tokenizer
            for term in terms:
                if term not in self.__term_idfs:
                    self.__term_idfs[term] = 1
                else:
                    self.__term_idfs[term] += 1
        #  Convert df's into idf's (inverted df)
        for term in self.__term_idfs.keys():
            self.__term_idfs[term] = log10(self.__size / self.__term_idfs[term])
        # Add the Theta as an additional element in the vectors
        self.__term_idfs['THETA'] = 1
        # Add list of features to corpus
        self.__corpus.set_all_feature_names(self.__term_idfs.keys())


    def get_tf_idf(self, tweet):
        """
        For each term in tweet calculate tf, normalized by tweet size.
        Calculate tf-idf scores using pre-calculated idf-scores.
        """
        # get term frequencies
        tfs = {}
        tokens = Tokenizer(tweet.get_text()).get_tokens()
        for token in tokens:
            if token in tfs.keys():
                tfs[token] += 1
            else:
                tfs[token] = 1
        # normalize by token count
        for term in tfs.keys():
            tfs[term] /= len(tokens)
        # calculate tf-idfs
        tf_idfs = {}
        for term in tfs.keys():
            tf_idfs[term] = tfs[term] * self.__term_idfs[term]
        # Add the Theta as an additional element in the vectors
        tf_idfs['THETA'] = 1

        return tf_idfs


    def set_features(self):
        """
        Calculate features (a vector of tf-idf scores) for each tweet and send
        to its corresponding Tweet object.
        """
        for tweet in self.__corpus:
            features = self.get_tf_idf(tweet)
            tweet.set_features(features)
