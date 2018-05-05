from tokenizer import Tokenizer
from math import log10

class Featurer():

    def __init__(self, corpus):
        self.corpus = corpus
        self.size = 0  # corpus size, i.e. number of tweets
        self.term_idfs = {} # key: term, value: idf-score
        self.get_idf_scores()  # calculate idf


    def __iter__(self):
        self.__curr = 0
        self.tweets = iter(self.corpus)
        return self


    def __next__(self):
        if self.__curr == self.size:
            raise StopIteration
        else:
            self.__curr += 1
            tweet = next(self.tweets)
            label = tweet.get_gold_label()
            features = self.get_tfidf(tweet)
            text = tweet.get_text()
            return (label, text, features)


    def get_idf_scores(self):

        #  First calculate document frequencies, i.e. the number of docs where
        #  the term occurs. At the same time we'll get the collection size.
        for tweet in self.corpus:
            self.size += 1
            terms = Tokenizer(tweet.get_text()).get_terms()  # TODO: adapt to Tokenizer
            for term in terms:
                if term not in self.term_idfs:
                    self.term_idfs[term] = 1
                else:
                    self.term_idfs[term] += 1

        #  Now, convert df's into idf's (inverted df)
        for term in self.term_idfs.keys():
            self.term_idfs[term] = log10(self.size / self.term_idfs[term])


    def get_tfidf(self, tweet):
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
        return self.term_idfs.keys()
