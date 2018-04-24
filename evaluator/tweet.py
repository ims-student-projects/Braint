class Tweet(object):

    def __init__(self, text, gold_label=None, pred_label=None):
        self.__text = text
        self.__pred_label = pred_label
        self.__gold_label = gold_label

    def get_text(self):
        return self.__text

    def get_pred_label(self):
        return self.__pred_label

    def get_gold_label(self):
        return self.__gold_label

    def set_pred_label(self, pred_label):
        self.__pred_label = pred_label

    def __str__(self):
        return ("Gold Label:" + "\t" + self.get_gold_label() + "\n" +
                "Predicted Label:" + "\t" + self.get_pred_label() + "\n" +
                "Text:" + "\t" + self.get_text()) 
