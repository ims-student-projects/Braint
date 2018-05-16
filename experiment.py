from Evaluator.scorer import Scorer
from Evaluator.result import Result
from corpus import Corpus
from featurer import Featurer
#from baseline_feature_extractor import FeatureExtractor
from multiclass_perceptron import MulticlassPerceptron

def main():
    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']

    # TRAIN DATA
    #pred_with_tweets = '../data/sample/trial.csv.train_90' # predicted labels + tweet text
    #gold = '../data/sample/trial.label.train_90' # file contains gold labels
    #train_corpus = Corpus(pred_with_tweets, gold)
    gold_with_tweets = '../data/train.csv'
    train_corpus = Corpus(gold_with_tweets)

    # TEST DATA
    #test_tweets = '../data/sample/trial.csv.test_10' # predicted labels + tweet text
    #gold_labels = '../data/sample/trial.label.test_10'
    #test_corpus = Corpus(test_tweets, gold_labels)
    test_tweets = '../data/trial.csv'
    gold_labels = '../data/trial.labels'
    test_corpus = Corpus(test_tweets, gold_labels)

    # Extract features
    #feature_extractor = FeatureExtractor()
    #feature_extractor.extract_features(train_corpus)
    #feature_extractor.extract_features(test_corpus)
    feature_extractor = Featurer(train_corpus)
    feature_extractor.set_features()
    #for tweet in train_corpus:
    #    print(tweet.get_features())
    feat_ex = Featurer(test_corpus)
    feat_ex.set_features()

    # Train classifier
    classifier = MulticlassPerceptron(classes, train_corpus.get_all_feature_names())
    classifier.train(1, train_corpus)
    # Test classifier
    classifier.test(test_corpus)

    # Evaluate test set
    scores = Scorer(test_corpus)
    result = Result()
    result.show(scores)

if __name__ == "__main__":
    main()
