import string

class Tokenizer():
    def __init__(self):
        self.punct = string.punctuation

    def get_tokens(self, text):
        tokens = text.split()
        result = []
        for token in tokens:
            if token.isalpha():
                result.append(token.lower())
            else:
                new_token=""
                for char in token:
                    if char in self.punct:
                        result.append(new_token.lower())
                        result.append(char)
                        new_token=""
                    else :
                        new_token +=char
                if new_token:
                    result.append(new_token.lower())
        return result


    def get_terms(self, text):
        unique = set(self.get_tokens(text))
        return list(unique)
