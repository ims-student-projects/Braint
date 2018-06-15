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

    # Configuration
    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']
    #types = [ 'binary', 'count', 'frequency', 'tf-idf']
    types = ['bigram']
    iterations = 150
    train_data = 'data/train'
    test_data = 'data/test'

    # Print some info
    bold = '\033[1m'
    unbold = '\033[0m'
    print_braint()
    print('{}Preparing to run Braint{}, using {} feature type(s), {} iterations each.'.format(
        bold, unbold, len(types), iterations))

    # Initiate corpora
    train_corpus = Corpus(train_data)
    test_corpus = Corpus(test_data)

    # Initiate feature extractors
    features_train = Featurer(train_corpus, bigram=True)
    features_test = Featurer(test_corpus, bigram=True)

    # Extract features of each type
    for type in types:
        print('Extracting features {}{}{}. Extracting features... '.format(bold, type, unbold))
        features_train.extract(type)
        features_test.extract(type)

        # Filenames used to save data
        fn_weights = 'experiment_{}_weights_lr0_05'.format(type)
        fn_scores = 'experiment_{}_scores_lr0_05'.format(type)

        # Create and train the model
        print('Training and testing model...')
        classifier = MulticlassPerceptron(classes, train_corpus.get_all_feature_names())
        classifier.train_and_test(iterations, train_corpus, test_corpus, fn_weights, fn_scores)


def print_braint():
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


if __name__ == "__main__":
    main()
