
class Scores():

    def __init__(self, iterator):
        self.tweets = iterator
        self.emotions = ['surprise', 'disgust', 'fear', 'sad', 'joy', 'anger']

        # Initiate base scores with values set to 0
        self.true_positives = {x:0 for x in self.emotions}
        self.false_positives = {x:0 for x in self.emotions}
        self.false_negatives = {x:0 for x in self.emotions}

        # Calculate the base scores
        self.get_base_scores()

        # F-score for each class in a dict
        self.f_all = {x:self.get_f_score(x) for x in self.emotions}

        # Micro and Macro F-scores
        self.f_micro = self.get_micro()
        self.f_macro = self.get_macro()


    def get_base_scores():
        """
        Calculate TP, FP and FN scores from the tweets.
        """
        for tweet in self.tweets:
            for emotion in self.emotions:
                if tweet.get_pred_label() == emotion:
                    if tweet.get_gold_label() == emotion:
                        true_positives[emotion] += 1
                    else:
                        false_positives[emotion] += 1
                elif tweet.get_gold_label() == emotion:
                    fasel_negatives[emotion] += 1


    def get_f_score(emotion):
        tp = self.true_positives[emotion]
        fp = self.false_positives[emotion]
        fn = self.false_negatives[emotion]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (2 * precision * recall ) / (precision + recall)


    def get_micro():
        tp = sum(self.true_positives.values())
        fp = sum(self.false_positives.values())
        fn = sum(self.false_negatives.values())

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (2 * precision * recall ) / (precision + recall)


    def get_macro():
        return sum(self.f_all.values()) / len(self.f_all)


class Result():
    def __init__(self, scores):
        pass
