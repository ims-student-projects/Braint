from corpus import Corpus
from scores import Scores

def main():

    gold = '../data/trial.labels' # file contains gold labels
    pred_with_tweets = '../data/trial.csv' # predicted labels + tweet text

    mycorpus = Corpus(pred_with_tweets, gold)
    myscores = Scores(mycorpus)
    print(myscores)

if __name__ == "__main__":
    main()
