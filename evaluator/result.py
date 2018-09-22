import matplotlib.pyplot as plt

class Result():
    """
    This class acts as an interface to display Evaluation scores on the terminal
    or write them into a file. A header is printed when the method show() is
    called for the first time. (header is the list of all scores that will be shown).

    Subesequently, each time the method show() is used, a new line of pr_scores
    will be printed. Nothing is printed if the write() method is called
    """

    def __init__(self):
        self.labels = [  'supP',  # Precision for Surprise
                         'supR',  # Recall for Surprise
                         'disP',  # Precision for Disgust
                         'disR',  # Recall for Disgust
                         'feaP',  # Precision for Fear
                         'feaR',  # Recall for Fear
                         'sadP',  # Precision for Sad
                         'sadR',  # Recall for Sad
                         'joyP',  # Precision for Joy
                         'joyR',  # Recall for Joy
                         'angP',  # Precision for Anger
                         'angR'   # Recall for Anger
                         ]
        self.print_header()
        self.convergence = []
        self.fmac = []
        self.epochs = 0


    def draw_graph(self, token_params, feature_type):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(self.convergence, label='Convergence (accuracy on train data)')
        ax.plot(self.fmac,  label='Macro F-score (test data)')
        ax.legend(loc='best')
        plt.axis([0, self.epochs, 0, 1])
        tp = ', '.join([str(p) for p in token_params if token_params[p]]) if token_params else 'None'
        plt.title('Tokenizer params: {}'.format(tp))
        plt.suptitle('Braint. Epochs: {}. Feature type: {}'.format(self.epochs, feature_type), y=.95, fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Convergence / F-macro')
        plt.show()

    def print_header(self):
        bold = '\033[1m'
        unbold = '\033[0m'
        header = 'Conv\t{}Fmac\tFmic{}\t{}'.format(
            bold, unbold, '\t'.join([l for l in self.labels]))
        print(header)


    def show(self, score, accuracy=None):
        self.convergence.append(accuracy)
        self.fmac.append(score.f_macro)
        self.epochs += 1
        print(score.__str__(accuracy))


    def write(self, accuracy, score, filename):
        with open(filename, 'a') as f:
            f.write(score.__str__(accuracy, use_bold=False) + '\n')
