
from corpus import Corpus
from featurer import Featurer

def main():
    """
    This is a demonstration of how the class Featurer can be used in combination
    with two classes Corpus and Tweet.
    """

    # first we create a corpus which is populated by the two files below
    pred_with_tweets = 'data/trial.csv' # predicted labels + tweet text
    gold = 'data/trial.labels' # file contains gold labels
    corp = Corpus(pred_with_tweets, gold)

    # now we create Featurer and let it pupulate our corpus with features
    feat = Featurer(corp)
    feat.set_features()

    # we can get a list of all feature labels from the corpus
    all_features = corp.get_all_feature_names()
    print(all_features)

    # we can also iterate over corpus and get the features of each tweet
    for tweet in corp:
        print(tweet.get_features())

    # voilà!

if __name__ == "__main__":
    main()
