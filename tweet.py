class Tweet(object):
    """ A datastructure to represent a tweet object.
    """

    def __init__(self, text : str, gold_label:str=None, pred_label:str=None):
        """ Inits Tweet.

            Args:
                text: the text of this Tweet as a string.
                (optional) gold_label: the gold label for this tweet as a string.
                (optional) pred_label: the predicted label for this tweet as a string.
        """
        self.__text = text
        self.__pred_label = pred_label
        self.__gold_label = gold_label
        self.__features = {}

    def set_features(self, features):
        """ Set the features for this Tweet.

            Args:
                features: a dictionary which maps from featurenames (strings) to values
        """
        self.__features = features

    def get_features(self):
        """ Returns the features for this Tweet.

            Returns:
                a dictionary which maps from feature names (strings) to values
        """
        return self.__features

    def get_text(self):
        """ Gets the text of this tweet.

            Returns:
                The tweet text as a String.
        """
        return self.__text

    def get_pred_label(self):
        """ Gets the predicted label of this tweet.

            Returns:
                The predicted label as a String.
        """
        return self.__pred_label

    def get_gold_label(self):
        """ Gets the gold label of this tweet.

            Returns:
                The gold label as a String.
        """
        return self.__gold_label

    def set_pred_label(self, pred_label : str):
        """ Sets the predicted label of this tweet.

            Args:
                pred_label: the predicted label for this tweet as a string.
        """
        self.__pred_label = pred_label

    def __str__(self):
        """ Generates a string representation of this tewwt object.

            Returns:
                A string representation of this tweet object,
                containg the tweet text, the predicted label and the gold label.
        """
        #return ("Gold Label:" + "\t" + self.get_gold_label() + "\n" +
        #        "Predicted Label:" + "\t" + self.get_pred_label() + "\n" +
        #        "Text:" + "\t" + self.get_text())
        return None
