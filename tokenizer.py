import string, re, emoji
from utils.translate_emoticon import emoticon_to_label
from utils.translate_emoji import emoji_to_label
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class Tokenizer():

    """
    :Tokenizer:

    Converts a stream of characters into a list of word-label list. Words are
    either tokens (all tokens in the string) or terms (only unique words).
    String is split on whitespace, punctuation, emojis and emoticons.

    :Usage:
        tokenizer = Tokenizer()
        tokenizer.get_tokens(text, OPTIONS)   -  for all tokens
        tokenizer.get_terms(text, OPTIONS)   -  for unique terms

    :Options:
        All options are boolean and are by default False:
            -- lowercase
            -- stem
            -- replace_emojis
            -- replace_num
            -- remove_stopw
            -- remove_punct
    """

    def __init__(self):
        self.punct = string.punctuation
        self.stopwords = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')


    def get_tokens(self, text,
            lowercase = False,
            stem = False,
            replace_emojis = False,
            replace_num = False,
            remove_stopw = False,
            remove_punct = False ):

        # Convert to lowercase
        if lowercase:
            text = text.lower()

        # Codify emoticons as tokens
        for emoticon in emoticon_to_label.keys():
            if re.search(emoticon, text):
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
                result.append((token, 'word'))

            # Token is special token
            elif re.fullmatch('^\[#(.*)#\]$', token):
                if token.lower() == '[#triggerword#]':
                    result.append((token, 'triggerword'))
                else:
                    # should be emoticon
                    result.append((token[2:-2], 'emoticon'))

            # Token is removed url
            elif token.lower() == 'http://url.removed':
                result.append(('[#URL#]', 'url'))

            # Look for punctuation, emojis or whitespace
            # Ignore newline symbol
            elif token.lower() != '[newline]':
                new_token = ''
                for char in token:
                    # Character is punctuation
                    if char in self.punct:
                        if not remove_punct:
                            result.append((char, 'punctuation'))
                        # add preceding charachters to results, if any
                        if new_token:
                            result.append(self.add_new_token(token, replace_num))
                            new_token = ''
                    # Character is common emoji
                    elif char in emoji_to_label.keys():
                        if replace_emojis:
                            emoji_text = emoji_to_label[char] # ignore columns
                            result.append((emoji_text[1:-1], 'emoji'))
                        else:
                            result.append((char, 'emoji'))
                        # add preceding charachters to results, if any
                        if new_token:
                            result.append(self.add_new_token(token, replace_num))
                            new_token = ''
                    # Character is uncommon emoji
                    elif char in emoji.UNICODE_EMOJI.keys():
                        if replace_emojis:
                            emoji_text = emoji.demojize(char) # ignore columns
                            result.append((emoji_text[1:-1], 'emoji'))
                        else:
                            result.append((char, 'emoji'))
                        # add preceding charachters to results, if any
                        if new_token:
                            result.append(self.add_new_token(token, replace_num))
                            new_token = ''
                    # Character is alpha-numerical
                    else:
                        new_token += char
                if new_token:
                    result.append(self.add_new_token(token, replace_num))

        # Remove stopwords
        if remove_stopw:
            result = self.remove_stopwords(result)

        # Stem
        if stem:
            result = self.get_stems(result)

        return result


    def add_new_token(self, token, replace_num):
        if token.isalpha():
            return (token, 'word')
        elif token.isnumeric():
            if replace_num:
                return ('[#num#]', 'numeric')
            return (token, 'numeric')
        return (token, 'other')  # Should happen, but to be save


    def remove_stopwords(self, word_list):
        processed_word_list = []
        for word in word_list:
            if word[1] == 'word' and word[0].lower() not in self.stopwords:
                processed_word_list.append(word)
            elif word[1] != 'word':
                processed_word_list.append(word)
        return processed_word_list


    def get_terms(self, text,
            lowercase = False,
            stem = False,
            replace_emojis = False,
            replace_num = False,
            remove_stopw = False,
            remove_punct = False ):

        tokens = self.get_tokens(text, lowercase, stem, replace_emojis,
            replace_num, remove_stopw, remove_punct)

        unique_tokens = []  # list of strings
        terms = []  # list of tuples
        for token in tokens:
            if token[0] not in unique_tokens:
                unique_tokens.append(token[0])
                terms.append(token)
        return terms


    def get_stems(self, tokens):
        stemmed_tokens = []
        for token in tokens:
            if token[1] == 'word':
                new_token = self.stemmer.stem(token[0])
                if new_token:
                    stemmed_tokens.append((new_token, 'word'))
                else:
                    stemmed_tokens.append(token)
            else:
                stemmed_tokens.append(token)
        return stemmed_tokens


if __name__ == '__main__':

    tokenizer = Tokenizer()
    test_sentence = 'HeLlO\t,  WoRld! I\'m Tired of losers <33333 1984 :)))) [NEWLINE] >:\ 🤠 🙂 😃😄😆😍'

    tokens = tokenizer.get_tokens(test_sentence, stem=False, lowercase=False,
        remove_stopw = True, replace_emojis=False)

    print('Tuples tokens ({}):\n{}\n'.format(len(tokens), tokens))
    print('String tokens:'.format(len(tokens)))
    print('"{}"\n'.format('" "'.join(t[0] for t in tokens)))
