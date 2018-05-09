from operator import itemgetter

class MulticlassPerceptron(object):

    def __init__(self, classes, feature_names):
        self.classes = classes
        # dict of dicts: ("name of class" --> ("name of feature" --> weight))
        self.weights = {}
        for c in self.classes:
            self.weights[c] = {}
            for feature in feature_names:
                self.weights[c][feature] = 0 # TODO: initialize weights randomly

    def update_weights(self, features, prediction, true_label):
        # only update if prdiction was wrong
        if prediction != true_label:
            for feat in features:
                # increase weights of features of example in correct class as these are important for classification
                self.weights[true_label][feat] += 1
                # decrease weights of features of example in wrongly predicted class
                self.weights[prediction][feat] -= 1
    
    def predict(self, features):
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
        for i in range(num_iterations):
            for example in examples:
                true_label = example[0]
                tweet_features = example[2] # dict
                prediction = self.predict(tweet_features)
                self.update_weights(tweet_features, prediction[0], true_label)
    