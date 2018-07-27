from tokenizer import Tokenizer
from corpus import Corpus
from utils.progress_bar import print_progressbar
from math import log10
from  nltk import pos_tag

class Featurer():
    """
    :Featurer:

    Extracts features from tweet collection. Features can be both unigram and
    bigram tokens. Feature values can be calculated in 4 methods (see under
    Options). An additional feature, '<BIAS>' (whose value is always 1) is
    added to the features, since those are intended to be used by a Perceptron.

    Don't wait for any return values, features are sent to the Corpus and each
    corresponding Tweet instance

    :Paramteres:
        - corpus -- an instance of class Corpus
        - token_params -- parameters that will be sent to Tokenizer, dict
        - grams -- number of grams to extract: 1 or 2 or both, tuple of ints
        - type -- feature type, str, select one from:
            -- binary
            -- count
            -- frequency
            -- tf_idf

    :Output:
    Directly fed back into the corpus and its tweet objects. See the following
    methods to access the features:
        - corpus.get_all_features() -- list of all feature labels in corpus
        - tweet.get_features() -- dict of features and values for each tweet

    :Usage:
        f = Featurer(corpus, PARAMETERS)
    :Example usage:
        f = Featurer(train_corpus, token_params=None, grams=(1,2), type='frequency')

    """

    def __init__(self, \
                corp=None, \
                token_params=None, \
                grams=(1,), \
                type='frequency',
                pos=False):

        """
        Greacefully exit if arguments are invalid
        """
        invalid_arguments = []
        if not corp or not isinstance(corp, Corpus):
            invalid_arguments.append('corp ({})'.format(corp))
        for g in grams:
            if g not in (1,2,3):
                invalid_arguments.append('grams ({})'.format(grams))
        if type not in ('binary', 'count', 'frequency', 'tf_idf'):
            invalid_arguments.append('type ({})'.format(type))
        if invalid_arguments:
            raise ValueError('\nInvalid argument(s): {}\n{}'
                .format(','.join(invalid_arguments), self.__doc__))


        self.corpus = corp
        self.corpus_size = corp.length()
        self.token_params = token_params
        self.count_unigrams = True if 1 in grams else False
        self.count_bigrams = True if 2 in grams else False
        self.count_trigrams = True if 3 in grams else False
        self.type = type
        self.count_pos = pos
        self.feature_labels = {'<BIAS>'}
        self.extract()


    def extract(self):
        """
        Main routine the performs some basic calculations before feature vectors
        are extracted:
        (1) counts collection size,
        (2) extracts list of all terms from colection (a.k.a. "feature names")

        Extract features for each tweet and send to corresponding Tweet object.
        If no or invalid type is given as parameter, print a soft warning.
        """

        # Prepare parameters for feature extraction (only for convenience)
        (u,b,t) = (self.count_unigrams, self.count_bigrams, self.count_trigrams)
        bin = True if self.type == 'binary' else False
        count = True if self.type == 'count' else False
        freq = True if self.type == 'frequency' else False
        tf_idf = True if self.type == 'tf_idf' else False

        # Optionally calculate IDF scores
        if self.type == 'tf_idf':
            self.calculate_idf_scores(u,b,t)

        # Main job: extract features and print progressbar
        count = 0
        print_progressbar(count, self.corpus_size)
        for tweet in self.corpus:
            features = self.extract_features(tweet,u,b,t,bin,count,freq,tf_idf)
            tweet.set_features(features)
            count += 1
            print_progressbar(count, self.corpus_size)

        # Add feature labels to corpus
        self.corpus.set_all_feature_names(self.feature_labels)


    def extract_features(self, \
                        tweet, \
                        count_unigrams, \
                        count_bigrams, \
                        count_trigrams, \
                        binary, \
                        count, \
                        frequency, \
                        tf_idf ):

        features = {}

        if count_unigrams:
            tokens = Tokenizer().get_tokens(tweet.get_text(), **self.token_params)
            for token in tokens:
                if count_unigrams:
                    unigram = token[0]
                    features[unigram] = 1 if binary else features.get(unigram,0)+1
                    self.feature_labels.add(unigram)

        if count_bigrams or count_trigrams:
            tp = self.token_params.copy()
            tp['stem'] = False
            tp['replace_emojis'] = False
            tp['replace_num'] = False
            strict_tokens = Tokenizer().get_tokens(tweet.get_text(), **tp)

            previous_token = '<BEGIN>'  # Used for bigrams
            previous = '<BEGIN>'  # Used for trigrams
            previous_previous = None  # Used for trigrams

            for token in strict_tokens:
                if count_bigrams:
                    bigram = previous_token + ' ' + token[0]
                    previous_token = token[0]
                    features[bigram] = 1 if binary else features.get(bigram,0)+1
                    self.feature_labels.add(bigram)
                if count_trigrams:
                    if previous_previous:
                        trigram = previous_previous + ' ' + previous + ' ' + token[0]
                        features[trigram] = 1 if binary else features.get(trigram,0)+1
                        self.feature_labels.add(trigram)
                    previous_previous = previous
                    previous = token[0]

        if self.count_pos:
            plain_tokens = [t[0] for t in tokens]
            tokens_with_pos = pos_tag(plain_tokens)
            for token in tokens_with_pos:
                tag = '<' + token[1] + '>'
                features[tag] = 1 if binary else features.get(tag,0)+1
                self.feature_labels.add(tag)

        if frequency or tf_idf:
            for f in features:
                features[f] /= len(tokens)

        if tf_idf:
            for f in features:
                if self.feature_idf_scores[f]:
                    features[f] *= self.feature_idf_scores[f]

        features['<BIAS>'] = 1
        return features


    def calculate_idf_scores(self, count_unigrams, count_bigrams, count_trigrams):
        """
        Extracts terms from corpus and adds them to self.__term_idfs. Calculates
        df score (=document frequency, i.e. number of tweets in which the term
        occurs). Finally df's
        are converted to idf's (=inverted df, i.e. documents with low df get a
        higher score).
        """

        self.feature_idf_scores = {}
        corpus_size = self.corpus.length()

        # Count document frequencyt of each feature
        for tweet in self.corpus:
            features = set()
            if count_unigrams:
                terms = Tokenizer().get_terms(tweet.get_text(), **self.token_params)
                for term in terms:
                    features.add(term[0])
            if count_bigrams:
                tokens = Tokenizer().get_tokens(tweet.get_text(), **self.token_params)
                previous_token = '<BEGIN>'
                for token in tokens:
                    bigram = previous_token + ' ' + token[0]
                    previous_token = token[0]
                    features.add(bigram)
            if count_trigrams:
                tokens = Tokenizer().get_tokens(tweet.get_text(), **self.token_params)
                previous_token = '<BEGIN>'
                previous_previous = None
                for token in tokens:
                    if previous_previous:
                        trigram = previous_previous + ' ' + previous_token + ' ' + token[0]
                        features.add(trigram)
                    previous_previous = previous_token
                    previous_token = token[0]

            for f in features:
                self.feature_idf_scores[f] = self.feature_idf_scores.get(f,0)+1

        #  Convert df's into idf's (inverted document frequency)
        for f in self.feature_idf_scores.keys():
            self.feature_idf_scores[f] = log10(corpus_size / self.feature_idf_scores[f])
