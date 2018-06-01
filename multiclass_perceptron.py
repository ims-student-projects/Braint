from operator import itemgetter
from tweet import Tweet
import json

class MulticlassPerceptron(object):
    """ A perceptron for multiclass classification.
    """

    def __init__(self, classes, feature_names):
        """ Inits the MulticlassPerceptron.
            Args:
                classes: a list contaning the class names as strings
                feature_names: a list containing all names of the features that are used.
        """
        self.classes = classes
        # dict of dicts: ("name of class" --> ("name of feature" --> weight))
        self.weights = {}
        for c in self.classes:
            self.weights[c] = {}
            for feature in feature_names:
                self.weights[c][feature] = 0 # TODO: initialize weights randomly


    def __update_weights(self, features, prediction, true_label):
        """ Updates the weights of the perceptron.
            Args:
                features: an iterable containg the names of the features of the current example
                prediction: the predicted label as a string
                true_label: the true label as a string
        """
        if prediction != true_label: # only update if prdiction was wrong
            for feat in features:
                # increase weights of features of example in correct class as these are important for classification
                self.weights[true_label][feat] += 1
                # decrease weights of features of example in wrongly predicted class
                self.weights[prediction][feat] -= 1


    def __predict(self, features, example, weights=None):
        """ Returns a prediction for the given features. Calculates activation
            for each class and returns the class with the highest activation.
            Args:
                features: dictionary containing features and values for these
                example: tweet object for which prediction is made
            Returns:
                a tuple containg (predicted_label, activation)
        """
        activations = []
        # calculate activation for each class
        for c in self.classes:
            curr_activation = 0
            for feat in features:
                # necessary if test examples contain unseen features - is there a better way to handle this?
                if feat in self.weights[c]:
                    curr_activation += self.weights[c][feat] * features[feat]
            activations.append((c, curr_activation))
        # highest activation in activation[0]
        activations.sort(key=itemgetter(1), reverse=True)
        # set prediction in tweet
        example.set_pred_label(activations[0][0])
        return activations[0]


    def train(self, num_iterations, examples, fn_acc=None, fn_weights=None):
        """ Function to train the MulticlassPerceptron. Optionally writes weights
            and accuracy into files.
            Args:
                num_iterations: the number of passes trough the training data
                examples: corpus (iterable) containing Tweets
                fn_acc: file where to write accuracy scores for each iteration
                fn_acc: file where to write weights for each iteration
        """
        accuracies = []
        for i in range(num_iterations):
            corr = 0  # correct predictions
            for example in examples:
                true_label = example.get_gold_label()
                tweet_features = example.get_features() # dict
                prediction = self.__predict(tweet_features, example)
                self.__update_weights(tweet_features, prediction[0], true_label)
                # keep count of correct/incorrect predictions
                corr += 1 if true_label == prediction[0] else 0

            # calculate accuracy score for iteration
            acc = round((corr / examples.length()), 2)
            accuracies.append(acc)

            # if requested, write current weights into file
            if fn_weights:
                self.save_model(fn_weights)

        # if requested, write accuracy results to file
        if fn_acc:
            with open(fn_acc, 'w') as f:
                f.write('\n'.join([str(a) for a in accuracies]))


    def __debug_print_prediction(self, example, prediction):
        print("true label: " + example.get_gold_label())
        print("tweet text: " + example.get_text())
        print("prediction: " + example.get_pred_label())
        print(prediction)


    def test(self, examples, weights=None):
        """
        Will use custom weights is passed by argument, otherwise class weights
        will be used.

        Args:
        examples: corpus (iterable) containing Tweets
        weights: (optional) use custom weights
        """
        # reset class weights if custom weights are provided
        if weights:
            self.load_model(weights)

        for example in examples:
            tweet_features = example.get_features() # dict
            prediction = self.__predict(tweet_features, example, weights)


    def save_model(self, filename):
        with open(filename, 'a') as w:
            w.write(json.dumps(self.weights) + '\n')


    def load_model(self, weights):
        self.weights = weights
