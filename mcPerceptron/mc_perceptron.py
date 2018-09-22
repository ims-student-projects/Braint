from operator import itemgetter
from tweet import Tweet
from evaluator.result import Result
from evaluator.scorer import Scorer
import json

class mcPerceptron(object):
    """
    A perceptron for multiclass classification.

        Args:
            classes: a list contaning the class names as strings
            feature_names: a list containing all names of the features that are used.
    """

    def __init__(self, classes:list, feature_names:set, parameters:dict, token_options:dict):
        self.parameters = parameters
        self.token_options = token_options
        self.lr = parameters['learning rate']
        self.classes = classes  # Names of emotions
        # Initialize weights as dict of dicts: ("class" --> ("feature" --> weight))
        self.weights = {c:{f:0 for f in feature_names} for c in classes}
        self.averaged_weights = {c:{f:0 for f in feature_names} for c in classes}
        self.num_steps = 0  # Used to average weights
        self.curr_step = 0  # Used to average weights


    def __update_weights(self, features, prediction, true_label):
        """ Updates the weights of the perceptron.
            Args:
                features: an iterable containg the names of the features of the current example
                prediction: the predicted label as a string
                true_label: the true label as a string
        """
        if prediction != true_label: # only update if prediction was wrong
            # Calculate rate for averaging
            r = (self.curr_step / self.num_steps) if self.curr_step != 0 else 0
            for feat in features:
                z = (features[feat] * self.lr)
                avg_z = (r*z)
                # increase weights of features of example in correct class as these are important for classification
                self.weights[true_label][feat] += z
                self.averaged_weights[true_label][feat] += avg_z
                # decrease weights of features of example in wrongly predicted class
                self.weights[prediction][feat] -= z
                self.averaged_weights[prediction][feat] -= avg_z

    def __predict(self, features, example, test_mode=False):
        """ Returns a prediction for the given features. Calculates activation
            for each class and returns the class with the highest activation.
            Args:
                features: dictionary containing features and values for these
                example: tweet object for which prediction is made
            Returns:
                a tuple containg (predicted_label, activation score)
        """
        weights = self.averaged_weights if test_mode else self.weights
        activations = []
        # calculate activation for each class
        for c in self.classes:
            curr_activation = 0
            for feat in features:
                # necessary if test examples contain unseen features - is there a better way to handle this?
                if feat in weights[c]:
                    curr_activation += weights[c][feat] * features[feat]
            activations.append((c, curr_activation))
        # highest activation in activation[0]
        activations.sort(key=itemgetter(1), reverse=True)
        # set prediction in tweet
        example.set_pred_label(activations[0][0])
        return activations[0]


    def train_and_test(self, train_corpus, test_corpus):

        result = Result() if self.parameters['print results'] else None

        self.train(train_corpus, test_corpus, result)

        if self.parameters['print plot']:
            result.draw_graph(self.token_options, type)

        if self.parameters['save test predictions']:
            with open(self.parameters['save test predictions'], 'w') as f:
                for tweet in test_corpus:
                    f.write(tweet.get_pred_label() + "\n")


    def train(self, train_corpus, test_corpus=None, result=None):
        """ Function to train the MulticlassPerceptron. Optionally writes weights
            and accuracy into files.
            Args:
                num_iterations: the number of passes trough the training data
                examples: corpus (iterable) containing Tweets
                fn_acc: file where to write accuracy scores for each iteration
                fn_acc: file where to write weights for each iteration
                # TODO
        """
        epochs = self.parameters['epochs']
        self.num_steps = epochs * train_corpus.length()
        self.curr_step = self.num_steps
        acc = 0  # accuracy score
        for i in range(epochs):
            corr = 0  # correct predictions during current iteration
            train_corpus.shuffle()  # shuffle tweets
            for tweet in train_corpus:
                true_label = tweet.get_gold_label()
                tweet_features = tweet.get_features() # dict
                prediction = self.__predict(tweet_features, tweet)
                self.__update_weights(tweet_features, prediction[0], true_label)
                self.curr_step -= 1
                # Count of correct predictions
                corr += 1 if true_label == prediction[0] else 0

            # Calculate accuracy score for current iteration
            # This score shows how the model is converging
            acc = round((corr / train_corpus.length()), 2)

            # Test on current weights
            if test_corpus:
                self.test(test_corpus, test_mode=True)
                scores = Scorer(test_corpus)

                if self.parameters['print results']:
                    result.show(scores, acc)

                if self.parameters['save results']:
                    result.write(acc, scores, self.parameters['save results'])

        # Write final weights to file
        if self.parameters['save model']:
            self.save_model()


    def __debug_print_prediction(self, example, prediction):
        print("true label: " + example.get_gold_label())
        print("tweet text: " + example.get_text())
        print("prediction: " + example.get_pred_label())
        print(prediction)


    def test_model(self, test_corpus, weights, output=None, evaluate=False):

        self.load_model(weights)
        self.test(test_corpus)

        if output:
            with open(output, 'w') as outf:
                for tweet in test_corpus:
                    outf.write(tweet.get_pred_label() + "\n")

        if evaluate:
            result = Result()
            scores = Scorer(test_corpus)
            result.show(scores,0)


    def test(self, test_corpus, test_mode=False):
        """
        Will use custom weights is passed by argument, otherwise class weights
        will be used.

        Args:
        examples: corpus (iterable) containing Tweets
        weights: (optional) use custom weights
        """
        # reset class weights if custom weights are provided
        #if weights:
        #    self.load_model(weights)

        for tweet in test_corpus:
            prediction = self.__predict(tweet.get_features(), tweet, test_mode)


    def save_model(self, filename=None):
        f = filename if filename else self.parameters['save model']
        with open(f + '_averaged', 'a') as w:
            w.write(json.dumps(self.averaged_weights) + '\n')
        with open(f, 'a') as w:
            w.write(json.dumps(self.weights) + '\n')


    def load_model(self, weights):
        with open(weights, 'r') as w:
            self.weights = json.load(w)
