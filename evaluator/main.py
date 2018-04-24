import corpus

def main():
    file_tweets = "/Users/marina/Documents/Master/2018_SS/TeamLab/iest/trial.csv"
    file_labels = "/Users/marina/Documents/Master/2018_SS/TeamLab/iest/trial.labels"
    corpus = Corpus.Corpus(file_tweets, file_labels)
    print(next(corpus))
    print(next(corpus))
    print(next(corpus))


if __name__ == "__main__":
    main()