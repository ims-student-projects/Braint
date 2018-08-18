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
        self.punct = string.punctuation + '’…”“£—＠→♡°⁎«＼｜／'
        self.stopwords = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')


    def get_tokens(self, text,
            lowercase = False,
            stem = False,
            replace_emojis = False,
            replace_num = False,
            remove_stopw = False,
            remove_punct = False,
            addit_mode = True ):

        # Convert to lowercase
        if lowercase:
            text = text.lower()

        # Codify emoticons as tokens
        for emoticon in emoticon_to_label.keys():
            if re.search(emoticon, text):
                if replace_emojis and addit_mode:
                    # Replace with emoticon + label (eg ":-)" > "[#:-)#] joy")
                    replace_by = '[#{}#] {}'.format(emoticon_to_label[emoticon][1:-1], emoticon)
                elif replace_emojis:
                    # Replace emoticon by label
                    replace_by = '[#{}#]'.format(emoticon_to_label[emoticon][1:-1])
                else:
                    # Just format emoticon
                    replace_by = '[#{}#]'.format(emoticon)
                text = re.sub(emoticon, replace_by, text)

        # Replace newline symbols with whitespace
        if re.search('[NEWLINE]', text):
            text = re.sub('\[NEWLINE\]', ' ', text)

        # Split string and analyze each token seperately
        result = []
        tokens = text.split()
        for token in tokens:

            # Token is word
            if token.isalpha():
                result.append((token, 'word'))

            # Token is special symbol, eg [#SYMBOL#]
            elif re.fullmatch('^\[#(.*)#\]$', token):
                if token.lower() == '[#triggerword#]':
                    result.append(('<TRIGGERWORD>', 'triggerword'))
                else:
                    # should be emoticon
                    result.append((token[2:-2], 'emoticon'))

            # Token is removed url
            elif token.lower() == 'http://url.removed':
                result.append(('<URL>', 'url'))

            # Look for punctuation, emojis or whitespace
            else:
                new_token = ''
                for char in token:
                    # Character is punctuation
                    if char in self.punct:
                        # add preceding charachters to results, if any
                        if new_token:
                            result += self.add_new_token(new_token, replace_num, stem, addit_mode)
                            new_token = ''
                        # Add symbol to result, except # and @
                        if char == '#' or char == '@':
                            new_token += char
                        elif not remove_punct:
                            result.append((char, 'punctuation'))
                    # Character is common emoji
                    elif char in emoji_to_label.keys():
                        # add preceding charachters to results, if any
                        if new_token:
                            result += self.add_new_token(new_token, replace_num, stem, addit_mode)
                            new_token = ''
                        # add emoji to results
                        if replace_emojis:
                            emoji_text = emoji_to_label[char] # ignore columns
                            result.append((emoji_text[1:-1], 'emoji'))
                            if addit_mode:
                                result.append((char, 'emoji'))
                        else:
                            result.append((char, 'emoji'))
                    # Character is uncommon emoji
                    elif char in emoji.UNICODE_EMOJI.keys():
                        # add preceding charachters to results, if any
                        if new_token:
                            result += self.add_new_token(new_token, replace_num, stem, addit_mode)
                            new_token = ''
                        # add emoji
                        if replace_emojis:
                            emoji_text = emoji.demojize(char) # ignore columns
                            result.append((emoji_text[1:-1], 'emoji'))
                            if addit_mode:
                                result.append((char, 'emoji'))
                        else:
                            result.append((char, 'emoji'))
                    # Character is alpha-numerical
                    else:
                        new_token += char
                # final check after loop
                if new_token:
                    result += self.add_new_token(new_token, replace_num, stem, addit_mode)
        # Remove stopwords
        if remove_stopw:
            result = self.remove_stopwords(result)
        return result


    def add_new_token(self, token, replace_num, stem, addit):
        """
        Returns a list with 1 or 2 elements: tuple with the new token and the
        label of the token (type of token). If the arguments replace_num or stem
        are True, additional token elements are added.
        """
        new_tokens = []
        if token[0] == '#':
            new_tokens.append((token, 'hashtag'))
        elif token[0] == '@':
            new_tokens.append((token, 'username'))
        elif token.isalpha():
            if stem:
                stemmed_token = self.stemmer.stem(token)
                if stemmed_token and stemmed_token != token:
                    new_tokens.append((stemmed_token, 'word'))
                    if addit:
                        new_tokens.append((token, 'word'))
                else:
                    new_tokens.append((token, 'word'))
        elif token.isnumeric():
            if replace_num:
                new_tokens.append(('<NUM>', 'numeric'))
                if addit:
                    new_tokens.append((token, 'numeric'))
            else:
                new_tokens.append((token, 'numeric'))
        else:
            new_tokens.append((token, 'other'))  # Shouldn't happen, but to be save
        return new_tokens


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
    test_1 = 'HeLlO\t,  WoRld! I\'m Tired of lo/sers <33333 1984 :)))) [NEWLINE] >:\ 🤠 🙂 😃😄😆😍'
    test_2 = "much♡[NEWLINE]•2 … …texting&driving he's @USERNAME works. A[NEWLINE][NEWLINE]As Mom:\"its pretty done."
    test_3 = '#WeLoveYouJackson[NEWLINE]#ItsOnlyGOT7'
    test_4 = '#Love#Love @user'
    test_5 = '’…”“£—＠→♡°⁎«＼｜／'
    test_6 = 'fu*kers'
    choice = test_1
    tokens = tokenizer.get_tokens(choice,
                lowercase=False,
                stem=True,
                replace_emojis=True,
                replace_num=True,
                remove_stopw=False,
                remove_punct=False,
                addit_mode=False)

    print('Original input: {}\n'.format(choice))
    print('Tuples tokens ({}):\n{}\n'.format(len(tokens), tokens))
    print('String tokens:'.format(len(tokens)))
    print('"{}"\n'.format('" "'.join(t[0] for t in tokens)))
