import string

class Tokenizer():
    def __init__(self):
        self.punct = string.punctuation

    def get_tokens(self, text):
        tokens = text.split()
        result = []
        for token in tokens:
            if token.isalpha() or token == '[#TRIGGERWORD#]' or token == 'http://url.removed':
                result.append(token)
            elif token != '[NEWLINE]': #ignote this newline thing
                new_token=""
                for char in token:
                    if char in self.punct:
                        result.append(char)
                        if new_token: # only add if not empty!
                            result.append(new_token)
                        new_token=""
                    else:
                        new_token +=char
                if new_token:
                    result.append(new_token)
        return result

    def remove_stopwords(self, word_list):
        processed_word_list = []
        for word in word_list:
            word = word.lower() # in case they arenot all lower cased
            # TODO if all are capital keep as itis and stopwords
            """if word not in stopwords.words("english"):
                processed_word_list.append(word)
        return processed_word_list"""
            processed_word_list.append(word)
        return  processed_word_list

    def get_terms(self, text):
        unique = set(self.get_tokens(text))
        return list(unique)

if __name__ == '__main__':
    a = Tokenizer()
    b = a.get_tokens("HeLlO\t,  WoRld!") # returned empty strings in previous version
    c = a.remove_stopwords(b)
    print(c)
