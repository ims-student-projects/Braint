from tokenizer import Tokenizer
from math import log10

class Featurer():
    """
    Extracts feature vectors for tweets in a corpus. Features are terms in
    Tweets represented by their tf-idf score. Expected input is an object
    of type Corpus. This class also uses the class Tokenizer to convert
    tweet text into tokens/terms.

    Featurer can be used as a generator to iterate over tweet features.
    A list of all features (i.e. terms in corpus) can be accessed by the
    method get_all_features().
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.size = 0  # corpus size, i.e. number of tweets
        self.term_idfs = {} # key: term, value: idf-score
        self.get_idf_scores()  # calculate idf


    def __iter__(self):
        """
        Make object iterable
        """
        self.__curr = 0
        self.tweets = iter(self.corpus)
        return self


    def __next__(self):
        """
        Return a tuple of label (emotion), tweet text and features (tf-dfs)
        """
        if self.__curr == self.size:
            raise StopIteration
        else:
            self.__curr += 1
            tweet = next(self.tweets)
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
        for tweet in self.corpus:
            self.size += 1
            terms = Tokenizer(tweet.get_text()).get_terms()  # TODO: adapt to Tokenizer
            for term in terms:
                if term not in self.term_idfs:
                    self.term_idfs[term] = 1
                else:
                    self.term_idfs[term] += 1

        #  Convert df's into idf's (inverted df)
        for term in self.term_idfs.keys():
            self.term_idfs[term] = log10(self.size / self.term_idfs[term])


    def get_tf_idf(self, tweet):
        """
        For each term in tweet calculate tf, normalized by tweet size.
        Calculate tf-idf scores using pre-calculated idf-scores.
        """
        # get term frequencies
        tfs = {}
        tokens = Tokenizer(tweet.get_text()).get_tokens()  # TODO
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
            tf_idfs[term] = tfs[term] * self.term_idfs[term]
        return tf_idfs


    def get_all_features(self):
        """
        Returns a list of all features, i.e. all terms in corpus.
        """
        return self.term_idfs.keys()
