from operator import itemgetter

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
    
    def __predict(self, features):
        """ Returns a prediction for the given features.
            Calculate activation for each class and return the class with the highest activation.

            Args:
                features: dictionary containing features and values for these

            Returns:
                a tuple containg (predicted_label, activation)
        """
        activations = []
        # calculate activation for each class
        for c in self.classes:
            curr_activation = 0
            for feat in features:
                curr_activation += self.weights[c][feat] * features[feat]
            activations.append((c, curr_activation))
        # highest activation in activation[0]
        activations.sort(key=itemgetter(1), reverse=True)
        return activations[0]

    def train(self, num_iterations, examples):
        """ Function to train the MulticlassPerceptron.

            Args:
                num_iterations: the number of passes trough the training data
                examples: a list containg (true_label, tweet_text, dict:feature_names->values)
        """
        for i in range(num_iterations):
            for example in examples:
                true_label = example[0]
                tweet_features = example[2] # dict
                prediction = self.__predict(tweet_features)
                self.__update_weights(tweet_features, prediction[0], true_label)
                self.print_prediction(example, prediction)

    def print_prediction(self, example, prediction):
        print("true label: " + example[0])
        print("tweet text: " + example[1])
        print("prediction: ")
        print(prediction)

    def test(self, examples):
        pass

    def save_model(self):
        pass
    
    def load_model(self):
        pass

