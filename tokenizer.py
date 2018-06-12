import string, re, emoji
from utils.translate_emoticon import emoticon_to_label


class Tokenizer():
    def __init__(self):
        self.punct = string.punctuation

    def get_tokens(self, text, \
            lowercase = False,
            replace_emojis = False,
            remove_stopword = False,
            remove_punct = False ):

        # Convert to lowercase
        if lowercase:
            text = text.lower()

        # Codify emoticons as tokens
        for emoticon in emoticon_to_label.keys():
            if replace_emojis:
                # Replace with labels (eg ":-)" > "[#joy#]")
                replace_by = '[#{}#]'.format(emoticon_to_label[emoticon][1:-1])
            else:
                replace_by = '[#{}#]'.format(emoticon)
            text = re.sub(emoticon, replace_by, text)

        tokens = text.split()
        result = []

        for token in tokens:
            # Token is word
            if token.isalpha():
                result.append({'token':token, 'label':'word'})
            # Token is special token
            elif re.fullmatch('^\[#(.*)#\]$', token):
                if token.lower() == '[#triggerword#]':
                    result.append({'token':token, 'label':'triggerword'})
                else:
                    result.append({'token':token[2:-2], 'label':'emoticon'})
            # Token is removed url
            elif token.lower() == 'http://url.removed':
                result.append({'token':'[#URL#]', 'label':'url'})

            elif token.lower() != '[newline]': # ignore newline symbol

            # Look for punctuation, emojis or whitespace and split
                new_token = ''
                for char in token:
                    # Character is punctuation
                    if char in self.punct:
                        if not remove_punct:
                            result.append({'token':char, 'label':'punctuation'})
                        # add preceding charachters to results, if any
                        if new_token:
                            result.append({'token':new_token, 'label':'word'})
                            new_token = ''
                    # Character is emoji
                    elif char in emoji.UNICODE_EMOJI.keys():
                        if replace_emojis:
                            emoji_text = emoji.demojize(char) # ignore columns
                            result.append({'token': emoji_text[1:-1], 'label': 'emoji'})
                        else:
                            result.append({'token':char, 'label':'emoji'})
                        # add preceding charachters to results, if any
                        if new_token:
                            result.append({'token':new_token, 'label':'word'})
                            new_token = ''
                    # Character is alpha-numerical
                    else:
                        new_token += char
                if new_token:
                    result.append({'token':new_token, 'label':'word'})
        return result


    def remove_stopwords(self, word_list):
        processed_word_list = []
        for word in word_list:
            #word = word.lower() # in case they arenot all lower cased
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
    tokenizer = Tokenizer()
    test_sentence = 'HeLlO\t,  WoRld! <33333 :)))) [NEWLINE] >:\ 🤠 🙂 😃😄😆😍'
    tokens = tokenizer.get_tokens(test_sentence, lowercase=True, remove_punct=True)
    print(tokens, '\n')
    print("Number of tokens: {}. Raw tokens: \n".format(len(tokens)))
    print('"{}"\n'.format('" "'.join(t['token'] for t in tokens)))
