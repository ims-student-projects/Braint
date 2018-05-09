from corpus import Corpus
from scorer import Scorer
from result import Result

def main():

    pred_with_tweets = '../data/trial.csv' # predicted labels + tweet text
    gold = '../data/trial.labels' # file contains gold labels
    mycorpus = Corpus(pred_with_tweets, gold)
    myscores = Scorer(mycorpus)
    myresult = Result()
    myresult.show(myscores)


if __name__ == "__main__":
    main()
