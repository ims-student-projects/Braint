from Evaluator.scorer import Scorer
from Evaluator.result import Result
from corpus import Corpus
#from featurer import Featurer
from baseline_feature_extractor import FeatureExtractor
from multiclass_perceptron import MulticlassPerceptron

def main():
    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']

    # TRAIN DATA
    pred_with_tweets = '../data/trial.csv' # predicted labels + tweet text
    gold = '../data/trial.labels' # file contains gold labels
    train_corpus = Corpus(pred_with_tweets, gold)

    # TEST DATA
    test_tweets = '../data/trial.csv.test_10' # predicted labels + tweet text
    gold_labels = '../data/trial.label.test_10'
    test_corpus = Corpus(test_tweets, gold_labels)

    # Extract features
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features(train_corpus)
    feature_names = feature_extractor.get_all_feature_names()
    feature_extractor.extract_features(test_corpus)

    # Train classifier
    classifier = MulticlassPerceptron(classes, feature_names)
    classifier.train(10, train_corpus)

    # Test classifier
    classifier.test(test_corpus)

    # Evaluate test set
    scores = Scorer(test_corpus)
    result = Result()
    result.show(scores)

if __name__ == "__main__":
    main()
