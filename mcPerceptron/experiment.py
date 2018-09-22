import sys
from time import time

sys.path.append('../')

from corpus import Corpus
from featurer import Featurer
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
        """
        'train_data': 'data/train-v3.csv',
        'test_data':'data/test-text-labels.csv',
        """
        experiment_parameters = {
                    'train data': '../data/s_train',
                    'test data':'../data/s_test',
                    'epochs': 35,
                    'learning rate': 0.3,
                    'ngrams': (1,),
                    'score': 'frequency',
                    'count pos': False,
                    'load model': None,
                    'save model': None,
                    'save test predictions': None,
                    'save results': None,
                    'print results': True,
                    'print plot': True,
                    'print class distribution': False,
                    'print progressbar': True
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

        # Print info about parameters
        print('Starting BrainT with parameters:\n{}'.format('\n'.join([' ' \
            '{}:\t{}'.format(p,v) for p,v in zip(self.parameters.keys(), \
            self.parameters.values())])))

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
        # TODO adapt argument processing in Featurer
        print('\nExtracting features for TRAIN data:')
        features_train = Featurer(train_corpus, self.parameters, self.token_options)
        print('Extracting features for TEST data:')
        features_test = Featurer(test_corpus, self.parameters, self.token_options)

        print('Training and testing model...\n')

        # TODO adapt argument processing
        model = mcPerceptron(
                    self.classes, \
                    train_corpus.get_all_feature_names(), \
                    self.parameters, \
                    self.token_options
                    )
        model.train_and_test(train_corpus, test_corpus)

        print('\nFinalized prediction and evaluation')

        if self.parameters['save model']:
            print('Model saved as {}'.format(self.parameters['save model']))

        if self.parameters['save test predictions']:
            print('Predictions saved as {}'.format(self.parameters['save test predictions']))

        print('Total runtime: {} s.'.format(round(time()-begin),3))


    def train(self):
        print('This is your TRAIN method')

    def test(self):
        print('This is your TEST method')

    def test_demo(self):
        print('This is your TEST DEMO method')


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
    exp = Experiment('train and test')
