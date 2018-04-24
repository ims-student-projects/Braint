from time import time

class Scores():

    def __init__(self, iterator):
        start = time() # used to count runtime

        self.tweets = iterator
        self.labels = ['surprise', 'disgust', 'fear', 'sad', 'joy', 'anger']

        # Initiate base scores with values set to 0
        self.true_positives = {x:0 for x in self.labels}
        self.false_positives = {x:0 for x in self.labels}
        self.false_negatives = {x:0 for x in self.labels}

        # Calculate the base scores
        self.get_base_scores()

        # F-score for each class in a dict
        self.f_all = {x:self.get_f_score(x) for x in self.labels}

        # Micro and Macro F-scores
        self.f_micro = self.get_micro()
        self.f_macro = self.get_macro()

        # Other
        self.runtime = round(time()-start, 3)


    def get_base_scores(self):
        """
        Calculate TP, FP and FN scores from the tweets.
        """
        for tweet in self.tweets:
            for label in self.labels:
                if tweet.get_pred_label() == label:
                    if tweet.get_gold_label() == label:
                        self.true_positives[label] += 1
                    else:
                        self.false_positives[label] += 1
                elif tweet.get_gold_label() == label:
                    self.false_negatives[label] += 1


    def get_f_score(self, label):
        tp = self.true_positives[label]
        fp = self.false_positives[label]
        fn = self.false_negatives[label]

        precision = tp / (tp+fp) if (tp+fp)!=0 else 0
        recall = tp / (tp + fn) if (tp+fn)!=0 else 0

        if (precision+recall) == 0:
            return 0.0

        return (2*precision*recall) / (precision+recall)


    def get_micro(self):
        tp = sum(self.true_positives.values())
        fp = sum(self.false_positives.values())
        fn = sum(self.false_negatives.values())

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (2 * precision * recall ) / (precision + recall)

    def get_macro(self):
        return sum(self.f_all.values()) / len(self.f_all)


    def __str__(self):
        text = 'Evaluation completed in {} seconds.\n' \
            'Macro F-score: {}. Micro F-score: {}'.format(self.runtime,
            round(self.f_macro, 3), round(self.f_micro, 3))
        return text
