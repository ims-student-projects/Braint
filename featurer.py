from tokenizer import Tokenizer
from corpus import Corpus
from math import log10
from operator import itemgetter

class Featurer():
    """
    :Featurer:

    Extracts feature vectors from a corpus. Feature type can specified in the
    arguments (see under Options). Since the features are intended to be used
    by a Perceptron, we always add the additional feature "THETA" (value is 1).

    :Expected input:
        - corpus -- an instance of class Corpus

        when extract() is called:
        - feature_type -- type of features to be extracted

    :Output:
    Directly fed back into the corpus and tweets:
        - corpus.get_all_features() -- list of all feature labels in corpus
        - tweet.get_features() -- dict of features and values for each tweet

    :Usage:
        features = Featurer(corpus)
        features.extract(feature_type)

    :Options:
        feature_type:
            -- binary
            -- count
            -- frequency
            -- tf-idf
            -- bigram
    """

    def __init__(self, corp=None, bigram=False):

        """
        Greacefully exit if corpus is not provided
        """
        if not corp or not isinstance(corp, Corpus):
            raise ValueError('\nMissing or invalid argument:\'corpus\'\n{}'
                .format(self.__doc__))

        # Dict that matches feature type to its method
        self._types = { 'binary': self._extract_binary,
                        'count': self._extract_count,
                        'frequency': self._extract_frequency,
                        'tf-idf': self._extract_tf_idf,
                        'bigram': self._extract_bigram}
        self._corpus = corp                 # iterable collection of tweets
        self._size = 0                      # corpus size = nr of tweets
        self._term_idfs = {}                # dict with term-idf-score pairs
        self._main(bigram)                  # start the fun


    def _main(self, bigram):
        """
        Main routine the performs some basic calculations before feature vectors
        are extracted:
        (1) counts collection size,
        (2) extracts list of all terms from colection (a.k.a. "feature names"),
        (3) sorts these terms according to df
        """
        if bigram:
            self._extract_bigram_labels()
        else:
            self._extract_idf()  # gets us size, terms and idf scores


    def extract(self, type=None):
        """
        Extract features for each tweet and send to corresponding Tweet object.
        If no or invalid type is given as parameter, print a soft warning.
        """
        if not type or type not in self._types.keys():
            print('invalid feature type {}. Ignoring request.'.format(type))
        else:
            for tweet in self._corpus:
                 # Fabricator to use method matching with requested feature type
                features = self._types[type](tweet)
                tweet.set_features(features)


    def _extract_idf(self):
        """
        Extracts terms from corpus and adds them to self.__term_idfs. Calculates
        df score (=document frequency, i.e. number of tweets in which the term
        occurs). Finally df's
        are converted to idf's (=inverted df, i.e. documents with low df get a
        higher score).
        """

        # Count corpus size, extract terms and count df for each term
        term_dfs = {}
        for tweet in self._corpus:
            self._size += 1
            terms = Tokenizer().get_terms(tweet.get_text())
            for term in terms:
                if term[0] not in term_dfs:
                    term_dfs[term[0]] = 1
                else:
                    term_dfs[term[0]] += 1

        #  Convert df's into idf's (inverted df)
        for term in term_dfs.keys():
            self._term_idfs[term] = log10(self._size / term_dfs[term])

        # Add the Theta as an additional element in the vectors
        self._term_idfs['THETA'] = 1

        # Add list of features to corpus
        self._corpus.set_all_feature_names(self._term_idfs.keys())


    def _extract_bigram_labels(self):
        bigram_labels = {}
        for tweet in self._corpus:
            self._size += 1
            # options are for DEBUG
            tokens = Tokenizer().get_tokens(tweet.get_text(), stem=False, lowercase=False,
                remove_stopw=False, replace_emojis=True, replace_num=False)
            previous = '<BEGIN>'
            for token in tokens:
                bigram = previous + ' ' + token[0]
                previous = token[0]
                if bigram not in bigram_labels.keys():
                    bigram_labels[bigram] = None
        bigram_labels['THETA'] = None

        # Add list of features to corpus
        self._corpus.set_all_feature_names(bigram_labels.keys())
        with open('experiment_bigram_labels', 'w') as f:
            f.write('", "'.join(bigram_labels.keys()))


    def _extract_tf_idf(self, tweet):
        """
        For each term in tweet calculate tf, normalized by tweet size.
        Calculate tf-idf scores using pre-calculated idf-scores.
        """
        # Get count of each term in tweet
        term_tfs = {}
        tokens = Tokenizer().get_tokens(tweet.get_text())
        for token in tokens:
            if token[0] in self._term_idfs.keys():
                if token[0] in term_tfs:
                    term_tfs[token[0]] += 1
                else:
                    term_tfs[token[0]] = 1
        # Normalize by total number of tokens in tweet
        for term in term_tfs:
            term_tfs[term] /= len(tokens)
        # Calculate tf-idf scores by mutltiplying tf and idf
        tf_idfs = {}
        for term in term_tfs:
            tf_idfs[term] = term_tfs[term] * self._term_idfs[term]
        # Add Theta as an additional element in the vector
        # (Yes, this needs to be added in both idf and tf-idf dicts!)
        tf_idfs['THETA'] = 1

        return tf_idfs


    def _extract_binary(self, tweet):
        binaries = {}
        tokens = Tokenizer().get_tokens(tweet.get_text())
        for token in tokens:
            if token[0] in self._term_idfs.keys():
                if token[0] not in binaries:
                    binaries[token[0]] = 1
        binaries['THETA'] = 1
        return binaries


    def _extract_count(self, tweet):
        counts = {}
        tokens = Tokenizer().get_tokens(tweet.get_text())
        for token in tokens:
            if token[0] in self._term_idfs.keys():
                if token[0] in counts:
                    counts[token[0]] += 1
                else:
                    counts[token[0]] = 1
        counts['THETA'] = 1
        return counts


    def _extract_frequency(self, tweet):
        frequencies = {}
        tokens = Tokenizer().get_tokens(tweet.get_text())
        for token in tokens:
            if token[0] in self._term_idfs.keys():
                if token[0] in frequencies:
                    frequencies[token[0]] += 1
                else:
                    frequencies[token[0]] = 1
        for token in frequencies:
            frequencies[token] /= len(tokens)
        frequencies['THETA'] = 1
        return frequencies


    def _extract_bigram(self, tweet):
        bigrams = {}
        tokens = Tokenizer().get_tokens(tweet.get_text(), stem=False, lowercase=False,
            remove_stopw=False, replace_emojis=True, replace_num=False)
        previous = '<BEGIN>'
        for token in tokens:
            bigram = previous + ' ' + token[0]
            previous = token[0]
            if bigram not in bigrams:
                bigrams[bigram] = 1
            else:
                bigrams[bigram] = 1

        for bigram in bigrams:
            bigrams[bigram] /= len(tokens)

        bigrams['THETA'] = 1

        return bigrams
