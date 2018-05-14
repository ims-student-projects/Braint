from corpus import Corpus

class FeatureExtractor(object):

    def __init__(self):
        self.all_feature_names = set()
    
    def get_tokens(self, text: str):
        tokens = text.split()
        # TODO:
        # remove punctuation
        # remove non-alphabetical tokens
        # filter stop words
        # filter short words
        # filter infrequent words
        return tokens

    def bow_counts(self, text):
        bow = {}
        tokens = self.get_tokens(text)
        for token in tokens:
            self.all_feature_names.add(token)
            if token in bow:
                bow[token] += 1
            else:
                bow[token] = 1
        bow['THETA'] = 1
        self.all_feature_names.add('THETA')
        return bow
    
    def bow_binary(self, text):
        bow = {}
        tokens = self.get_tokens(text)
        for token in tokens:
            self.all_feature_names.add(token)
            if token not in bow:
                bow[token] = 1
        bow['THETA'] = 1
        self.all_feature_names.add('THETA')
        return bow
    
    def bow_freq(self, text):
        bow = {}
        tokens = self.get_tokens(text)
        for token in tokens:
            self.all_feature_names.add(token)
            if token in bow:
                bow[token] += 1
            else:
                bow[token] = 1
        for token in bow:
            bow[token] = bow[token] / len(tokens)
        bow['THETA'] = 1
        self.all_feature_names.add('THETA')
        return bow

    def extract_features(self, corpus):
        for tweet in corpus:
            #tweet.set_features(self.bow_binary(tweet.get_text()))
            #tweet.set_features(self.bow_counts(tweet.get_text()))
            tweet.set_features(self.bow_freq(tweet.get_text()))
        
    

    def get_all_feature_names(self):
        return list(self.all_feature_names)