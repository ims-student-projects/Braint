
class Scorer():

    def __init__(self, iterator):
        self.tweets = iterator
        self.labels = ['surprise', 'disgust', 'fear', 'sad', 'joy', 'anger']

        # Initiate base scores with values set to 0
        self.true_positives = {x:0 for x in self.labels}
        self.false_positives = {x:0 for x in self.labels}
        self.false_negatives = {x:0 for x in self.labels}
        self.precision = {}
        self.recall = {}

        # Calculate the base scores
        self.get_base_scores()

        # F-score for each class in a dict
        self.f_all = {x:self.get_f_score(x) for x in self.labels}

        # Micro and Macro F-scores
        self.f_micro = self.get_micro()
        self.f_macro = self.get_macro()


    def get_base_scores(self):
        """
        Calculate TP, FP, FN, Precision and Recall scores for all labels.
        Adds to the corresponding dictionaries, no return data.
        """
        # Calculate TP, FP and FN for all labels
        for tweet in self.tweets:
            for label in self.labels:
                if tweet.get_pred_label() == label:
                    if tweet.get_gold_label() == label:
                        self.true_positives[label] += 1
                    else:
                        self.false_positives[label] += 1
                elif tweet.get_gold_label() == label:
                    self.false_negatives[label] += 1

        # Calculate Precision and Recall for all lables
        for label in self.labels:
            tp = self.true_positives[label]
            fp = self.false_positives[label]
            fn = self.false_negatives[label]
            self.precision[label] = tp / (tp+fp) if (tp+fp)!=0 else 0.0
            self.recall[label] = tp / (tp + fn) if (tp+fn)!=0 else 0.0


    def get_f_score(self, label):
        """
        Returns F-scores for each label.
        """
        p = self.precision[label]
        r = self.recall[label]
        return (2*p*r)/(p+r) if (p+r)!=0 else 0.0


    def get_macro(self):
        """
        Returns an average from the F-scores of all labels.
        """
        return sum(self.f_all.values()) / len(self.f_all)


    def get_micro(self):
        """
        Returns F-score calculated from all TP, FP and FN scores.
        """
        tp = sum(self.true_positives.values())
        fp = sum(self.false_positives.values())
        fn = sum(self.false_negatives.values())
        p = tp / (tp+fp) if (tp+fp)!=0 else 0.0
        r = tp / (tp+fn) if (tp+fn)!=0 else 0.0
        return (2*p*r)/(p+r) if (p+r)!=0 else 0.0


    def __str__(self):
        """
        Fancy list of macro, micro f-scores and precision and recall for all labels
        """
        bold = '\033[1m'
        unbold = '\033[0m'
        header = '{}Fmac\tFmic{}\tsupP\tsupR\tdisP\tdisR\tfeaP\tfeaR\tsadP\tsadR\tjoyP\tjoyR\tangP\tangR\n'.format(bold, unbold)
        pr_scores = '\t'.join('{}\t{}'.format(round(self.precision[l],2), round(self.recall[l],2)) for l in self.labels)
        all_scores = '{}{}\t{}{}\t{}'.format(bold, round(self.f_macro,3), round(self.f_micro,3),
            unbold, pr_scores)
        return header + all_scores
