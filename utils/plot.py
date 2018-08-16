import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(predictions_file):
    y_pred = []
    y_true = []
    is_first = True
    with open(predictions_file) as f:
        for line in f:
            if not is_first:
                line = line.strip().split()
                y_true.append(line[0].strip())
                y_pred.append(line[1].strip())
            else:
                is_first = False
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(confusion_matrix, classes, title, normalize=False):
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], '.2f' if normalize else 'd'), horizontalalignment="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2. else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    predictions_file = "/Users/marina/Documents/Master/2018_SS/TeamLab/experiment_eval/predictions_lstm.csv"
    title = "LSTM"
    classes = ["joy", "anger", "sad", "surprise", "fear", "disgust"]
    confusion_matrix = create_confusion_matrix(predictions_file)
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes, title, True)
    plt.show()

if __name__ == "__main__":
    main()