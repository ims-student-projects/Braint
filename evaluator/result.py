
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
        self.header_printed = False


    def print_header(self):
        bold = '\033[1m'
        unbold = '\033[0m'
        header = '{}Fmac\tFmic{}\t{}'.format(
            bold, unbold, '\t'.join([l for l in self.labels]))
        print(header)


    def show(self, score):
        if (self.header_printed):
            print(score)
        else:
            self.print_header()
            self.header_printed = True
            print(score)


    def write(self, score, filename):
        with open(filename, 'a') as f:
            f.write(score.__str__(bolded=False))
