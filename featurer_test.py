
from evaluator.corpus import Corpus
from featurer import Featurer

def main():
    pred_with_tweets = 'data/trial.csv' # predicted labels + tweet text
    gold = 'data/trial.labels' # file contains gold labels
    c = Corpus(pred_with_tweets, gold)
    f = Featurer(c)
    #print(f.get_all_features())
    for x in f:
        print(x)


if __name__ == "__main__":
    main()
