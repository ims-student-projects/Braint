from evaluator.scorer import Scorer
from evaluator.result import Result
from corpus import Corpus
from featurer import Featurer
#from baseline_feature_extractor import FeatureExtractor
from multiclass_perceptron import MulticlassPerceptron
from time import time

def main():
    # TODO let Corpus generate this list
    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']

    # TRAIN DATA
    gold_with_tweets = 'data/train'
    train_corpus = Corpus(gold_with_tweets)

    # TEST DATA
    gold_labels = 'data/test'
    test_corpus = Corpus(gold_labels)

    # Extract features
    features_train = Featurer(train_corpus, 5)
    features_test = Featurer(test_corpus, 5)
    types = [ 'binary', 'count', 'frequency', 'tf-idf']
    for type in types:
        print('Extracting features -- {}'.format(type))
        features_train.extract(type)
        features_test.extract(type)
        result = Result()

        # Train and test model
        for i in range(1,10):
            # train
            classifier = MulticlassPerceptron(classes, train_corpus.get_all_feature_names())
            classifier.train(i, train_corpus)
            # test
            classifier.test(test_corpus)
            # evaluate and show result
            scores = Scorer(test_corpus)
            result.show(scores)

if __name__ == "__main__":
    main()
