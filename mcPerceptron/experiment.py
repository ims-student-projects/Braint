import sys
from time import time

sys.path.append('../')

from corpus import Corpus
from featurer import Featurer
from tweet import Tweet
from mc_perceptron import mcPerceptron


class Experiment(object):
    """
    A fancy demonstration of how Braint can be used. Change variables under
    Configuration to try it out differently.

    IMPORTANT: Don't forget to remove output files, if you already have run Braint.
    E.g: run `rm experiment_*` in your terminal
    """

    def __init__(self, mode, parameters=None, options=None):
        # Default values for required parameters
        experiment_modes = {
                    'train': self.train,
                    'train and test': self.train_and_test,
                    'test': self.test,
                    'test demo': self.test_demo,
                    }
        experiment_parameters = {
                    'train data': '../data/train-v3.csv',
                    'test data':'../data/test-text-labels.csv',
                    'epochs': 35,
                    'learning rate': 0.3,
                    'ngrams': (2,3),
                    'score': 'frequency',
                    'count pos': False,
                    'load model': 'freq_23g_35e',
                    #'load model': 'dummy_model',
                    'save model': 'freq_23g_35e',
                    'save test predictions': None,
                    'save results': None,
                    'print results': True,
                    'print plot': True,
                    'print class distribution': False,
                    'print progressbar': False
                    }
        token_options = {
                    'addit_mode':True,
                    'lowercase':False,
                    'stem':False,
                    'replace_emojis':False,
                    'replace_num':False,
                    'remove_stopw':False,
                    'remove_punct':False
                    }
        self.classes = [
                    'joy',
                    'anger',
                    'fear',
                    'surprise',
                    'disgust',
                    'sad'
                    ]
        # Verify experiment parameters
        if parameters:
            assert all([True if p in experiment_parameters else False \
                for p in parameters]), 'Invalid parameters. Only accepting:' \
                    '\n {}'.format(',\n '.join(experiment_parameters.keys()))
            self.parameters = parameters
            #TODO if not all params are provided, take from default
        else:
            self.parameters = experiment_parameters
        # verify tokenization options
        if options:
            assert all([True if o in token_options else False for o in options]), \
                'Invalid token parameters. Only accepting:\n {}'.format \
                    (',\n '.join(token_options.keys()))
            self.token_options = options
            #TODO if not all options are provided, take from default
        else:
            self.token_options = token_options
        # Verify mode is valid and run
        assert mode in experiment_modes, 'Unexpected mode "{}". Please' \
        ' choose from: [{}]'.format(mode, ', '.join(experiment_modes))
        experiment_modes[mode]()


    def train_and_test(self):
        begin = time()
        self.print_intro()

        # Initialize corpora
        if self.parameters['print class distribution']:
            print('Class distribution in TRAIN data:')
        # TODO move print to Corpus
        train_corpus = Corpus(self.parameters['train data'],self.parameters['print class distribution'])

        if self.parameters['print class distribution']:
            print('Class distribution in TEST data (predictions):')
        # TODO same
        test_corpus = Corpus(self.parameters['test data'],self.parameters['print class distribution'])

        print('\nTokenizing tweets with options:\n{}'.format('\n'.join([' ' \
            '{}:\t{}'.format(o,v) for o,v in zip(self.token_options.keys(), \
            self.token_options.values())])))

        # Extract features
        print('\nExtracting features from TRAIN data:')
        features_train = Featurer(train_corpus, self.parameters, self.token_options)
        print('Extracting features from TEST data:')
        features_test = Featurer(test_corpus, self.parameters, self.token_options)

        print('Training and testing model...\n')

        model = mcPerceptron(
                    self.classes, \
                    self.parameters, \
                    self.token_options, \
                    train_corpus.get_all_feature_names()
                    )
        model.train_and_test(train_corpus, test_corpus)

        print('\nFinalized prediction and evaluation.')

        if self.parameters['save model']:
            print('Model saved as {}'.format(self.parameters['save model']))

        if self.parameters['save test predictions']:
            print('Predicted labels saved as {}'.format(self.parameters['save test predictions']))

        if self.parameters['save results']:
            print('Evaluation results saved as {}'.format(self.parameters['save results']))

        print('Total runtime: {} s.'.format(round(time()-begin,3)))


    def train(self):
        begin = time()
        self.print_intro()

        # Initialize corpus
        if self.parameters['print class distribution']:
            print('Class distribution in TRAIN data:')
        train_corpus = Corpus(self.parameters['train data'],self.parameters['print class distribution'])

        print('\nTokenizing tweets with options:\n{}'.format('\n'.join([' ' \
            '{}:\t{}'.format(o,v) for o,v in zip(self.token_options.keys(), \
            self.token_options.values())])))

        # Extract features
        print('\nExtracting features from TRAIN data:')
        features_train = Featurer(train_corpus, self.parameters, self.token_options)

        print('\nTraining model...\n')

        model = mcPerceptron(
                    self.classes, \
                    self.parameters, \
                    self.token_options, \
                    train_corpus.get_all_feature_names()
                    )
        model.train(train_corpus)

        print('\nTraining model completed.')

        if self.parameters['save model']:
            print('Model saved as {}'.format(self.parameters['save model']))

        print('Total runtime: {} s.'.format(round(time()-begin,3)))


    def test(self):
        begin = time()
        self.print_intro()

        if self.parameters['print class distribution']:
            print('Class distribution in TEST data (predictions):')
        test_corpus = Corpus(self.parameters['test data'],self.parameters['print class distribution'])

        print('\nTokenizing tweets with options:\n{}'.format('\n'.join([' ' \
            '{}:\t{}'.format(o,v) for o,v in zip(self.token_options.keys(), \
            self.token_options.values())])))

        print('\nExtracting features from TEST data:')
        features_test = Featurer(test_corpus, self.parameters, self.token_options)

        model = mcPerceptron(
                    self.classes, \
                    self.parameters, \
                    self.token_options
                    )
        print('\nLoading model from file [{}].\n'.format(self.parameters['load model']))
        model.test_model(test_corpus)

        print('\nModel evaluation completed.')

        if self.parameters['save test predictions']:
            print('Predicted labels saved as {}'.format(self.parameters['save test predictions']))

        if self.parameters['save results']:
            print('Evaluation results saved as {}'.format(self.parameters['save results']))

        print('Total runtime: {} s.'.format(round(time()-begin,3)))


    def test_demo(self):
        self.model = mcPerceptron(
                    self.classes, \
                    self.parameters, \
                    self.token_options
                    )
        self.model.load_model()


    def predict(self, text):
        tweet = Tweet(text)
        featurer = Featurer(None, self.parameters, self.token_options)
        features = featurer.extract_features(tweet)
        emotion = self.model._predict(features, tweet)
        return emotion[0][0]


    def print_intro(self):
        # Print info about parameters
        print('Starting BrainT with parameters:\n{}'.format('\n'.join([' ' \
            '{}:\t{}'.format(p,v) for p,v in zip(self.parameters.keys(), \
            self.parameters.values())])))


    def print_braint(self, type, epochs):
        bold = '\033[1m'
        unbold = '\033[0m'
        braint = """
        88                                         8888888888888888
        88                               (8)       ##^^^^888^^^^^##
        88                                               888
        88,dPPYba,  8b,dPPYba, ,adPPYYba, 88 8b,dPPYba,  888
        88P'    "8a 88P'   "Y8 ""     `Y8 88 88P'   `"8a 888
        88       d8 88         ,adPPPPP88 88 88       88 888
        88b,   ,a8" 88         88,    ,88 88 88       88 888
        8Y"Ybbd8"'  88         `"8bbdP"Y8 88 88       88 888
        """
        print(braint)
        print('{}Preparing to run Braint{}, using {} feature type, with {} epochs.\n'.format(
                    bold, unbold, type, epochs))


if __name__ == "__main__":
    #exp1 = Experiment('train and test')
    text = 'i have to be punched in the face by my loneliness before i reach out to someone'
    exp = Experiment('test demo')
    #exp.predict(text)
