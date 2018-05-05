
class Tokenizer():

    def __init__(self, text):
        self.text = text

    def get_tokens(self):
        return self.text.split()

    def get_terms(self):
        unique = set(self.get_tokens())
        return list(unique)
