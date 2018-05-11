from corpus import Corpus
from featurer import Featurer
from multiclass_perceptron import MulticlassPerceptron

def main():

    pred_with_tweets = '../data/trial.csv' # predicted labels + tweet text
    gold = '../data/trial.labels' # file contains gold labels
    corpus = Corpus(pred_with_tweets, gold)
    feature_extractor = Featurer(corpus)
    classes = ['joy', 'anger', 'fear', 'surprise', 'disgust', 'sad']
    classifier = MulticlassPerceptron(classes, feature_extractor.get_all_features())

    classifier.train(1, feature_extractor)

if __name__ == "__main__":
    main()
