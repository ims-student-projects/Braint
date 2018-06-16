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

    # Feature types ('binary', 'count', 'frequency', 'tf-idf')
    types = ['frequency']
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
    learning_rate = 0.5

    # Print info
    print_braint(types, epochs)

    # Initialize corpora
    train_corpus = Corpus(train_data)
    test_corpus = Corpus(test_data)

    # Initialize feature extractors
    features_train = Featurer(train_corpus, token_params, bigram=False)
    features_test = Featurer(test_corpus, token_params, bigram=False)

    # Extract features of each type
    for type in types:
        print('Extracting features {}... '.format(type.upper()))
        features_train.extract(type)
        features_test.extract(type)

        # Filenames used to save data
        fn_weights = 'experiment_av-shfl5_{}_weights'.format(type)
        fn_scores = 'experiment_av-shfl5_{}_scores'.format(type)

        # Create and train the model
        print('Training and testing model...')
        classifier = MulticlassPerceptron(classes, train_corpus.get_all_feature_names(), \
                    learning_rate)
        classifier.train_and_test(epochs, train_corpus, test_corpus, fn_weights, \
                    fn_scores, token_params, [type])


def print_braint(types, epochs):
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
    print('{}Preparing to run Braint{}, using {} feature type(s), {} epochs each.'.format(
                bold, unbold, len(types), epochs))


if __name__ == "__main__":
    main()
