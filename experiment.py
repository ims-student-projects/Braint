from evaluator.scorer import Scorer
from evaluator.result import Result
from corpus import Corpus
from featurer import Featurer
from multiclass_perceptron import MulticlassPerceptron
import json

"""
A fancy demonstration of how Braint can be used. Change variables under
Configuration to try it out differently.

Don't forget to remove output files, if you already have run Braint. E.g:
`rm experiment_*` in your terminal
"""

def main():

    # Configuration
    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']
    types = [ 'binary', 'count', 'frequency', 'tf-idf']
    stopw = 0 # Percentage of words to be filtered from features
    iterations = 25
    train_data = 'data/train'
    test_data = 'data/test'

    # Print some info
    bold = '\033[1m'
    unbold = '\033[0m'
    print_braint()
    print('{}Preparing to run Brant{}, using {} feature types, {} iterations each.\n'.format(
        bold, unbold, len(types), iterations))

    # Initiate corpora
    train_corpus = Corpus(train_data)
    test_corpus = Corpus(test_data)

    # Initiate feature extractors
    features_train = Featurer(train_corpus, stopw)
    features_test = Featurer(test_corpus, 0)

    # Extract features of each type
    for type in types:
        print('\nExtracting features {}{}{}. Training Model... '.format(bold, type, unbold))
        features_train.extract(type)
        features_test.extract(type)

        # Filenames used to save data
        fn_acc = 'experiment_{}_accuracies'.format(type)
        fn_weights = 'experiment_{}_weights'.format(type)
        fn_fscores = 'experiment_{}_fscores'.format(type)

        # Create and train the model
        classifier = MulticlassPerceptron(classes, train_corpus.get_all_feature_names())
        classifier.train(iterations, train_corpus, fn_acc, fn_weights)
        result = Result()

        print('Testing Model. Results: ')
        # Test the model using saved weights for each iteration
        with open(fn_weights, 'r') as f:
            for w in f:
                weight = json.loads(w)
                classifier.test(test_corpus, weight)
                scores = Scorer(test_corpus)
                result.show(scores)
                result.write(scores, fn_fscores)


def print_braint():
    brain = """
    88                                         8888888888888888
    88                               (8)       ##^^^^888^^^^^##
    88                                               888
    88,dPPYba,  8b,dPPYba, ,adPPYYba, 88 8b,dPPYba,  888
    88P'    "8a 88P'   "Y8 ""     `Y8 88 88P'   `"8a 888
    88       d8 88         ,adPPPPP88 88 88       88 888
    88b,   ,a8" 88         88,    ,88 88 88       88 888
    8Y"Ybbd8"'  88         `"8bbdP"Y8 88 88       88 888
    """
    print(brain)


if __name__ == "__main__":
    main()
