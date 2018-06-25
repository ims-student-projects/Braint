from corpus import Corpus
from featurer import Featurer
from multiclass_perceptron import MulticlassPerceptron
import json

"""
A fancy demonstration of how Braint can be used. Change variables under
Configuration to try it out differently.

IMPORTANT: Don't forget to remove output files, if you already have run Braint.
E.g: run `rm experiment_*` in your terminal
"""

def main():

    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']

    # Feature types ('binary', 'count', 'frequency', 'tf-idf', 'bigram')
    type = 'bigram'
    train_data = 'data/train'
    test_data = 'data/test'

    # Tokenizer parameters
    token_params = { 'lowercase':False,
                    'stem':False,
                    'replace_emojis':False,
                    'replace_num':False,
                    'remove_stopw':False,
                    'remove_punct':False }

    # Perceptron parameters
    epochs = 150
    learning_rate = 0.3

    # Print info
    print_braint(type, epochs)

    # Initialize corpora
    print('Class distribution in TRAIN data:')
    train_corpus = Corpus(train_data, print_distr=True)
    test_corpus = Corpus(test_data, print_distr=False)

    # Initialize feature extractors
    features_train = Featurer(train_corpus, token_params, bigram=True)
    features_test = Featurer(test_corpus, token_params, bigram=True)
    print('Tokenizing tweets... Parameters: {}'.format
            (', '.join([p for p in token_params if token_params[p]])))

    # Extract features of each type
    print('Extracting features {}... '.format(type.upper()))
    features_train.extract(type)
    features_test.extract(type)

    # Filenames used to save data
    fn_weights = 'results/experiment_{}_weights'.format(type)
    fn_scores = 'results/experiment_{}_scores'.format(type)

    # Create and train the model
    print('Training and testing model...\n')
    classifier = MulticlassPerceptron(classes, train_corpus.get_all_feature_names(), \
                learning_rate)
    classifier.train_and_test(epochs, train_corpus, test_corpus, fn_weights, \
                fn_scores, token_params, [type])

    print('Finlized predictiona and evaluation.\nClass distribution in TEST data:')
    test_corpus.print_distr()



def print_braint(type, epochs):
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
    main()
