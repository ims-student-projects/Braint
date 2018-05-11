import string

class Tokenizer():
    def __init__(self):
        self.punct = string.punctuation
        
    def get_tokens(self, text):
        tokens = text.split()
        result = []
        for token in tokens:
            if token.isalpha():
                result.append(token)
            else: 
                new_token=""
                for char in token:
                    if char in self.punct:
                        result.append(new_token)
                        result.append(char)
                        new_token=""
                    else :
                        new_token +=char
            if new_token:
                result.append(new_token)
        return result
                        
    def get_terms(self, text):
        p =['~', '!', '#', '$', '%', '^', '&', '*', ',', '?','_','*']
        newStr =''
        i =0
        punc = None
        for i in range(len(text)):
            if text[i] in p:
                newStr = ' '
                newStr = text[i]
                punc =False
            else:
                newStr = text[i]
                punc = True 
        return newStr.split('  ')
    
if __name__ == '__main__':
    a = Tokenizer()
    b = a.get_tokens("hello, world!")
    print(b)
