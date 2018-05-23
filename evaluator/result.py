
class Result():

    def __init__(self):
        self.print_header()


    def print_header(self):
        bold = '\033[1m'
        unbold = '\033[0m'
        header = '{}Fmac\tFmic{}\tsupP\tsupR\tdisP\tdisR\tfeaP\tfeaR\tsadP\tsadR\tjoyP\tjoyR\tangP\tangR'.format(bold, unbold)
        print(header)


    def show(self, score):
        print(score)
