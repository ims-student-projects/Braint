"""
A script to create and plot confusion matrices.
Notes:
    - One or multiple inputfiles, titles and classnames can be given. Multiple values for one parameter have to be seperated by semicolons (;) and placed in between quotes.
        e.g. python3 plot.py -i "path/to/file1;path/to/file2" -t "model1;model2" -c "class1;class2;class3" -n 1
             python3 plot.py -i path/to/file -t model1 -c "class1;class2;class3" -n 1
    - If the value of -n is 1 the confusion matrix values are normalized otherwise not.
    - The inputfiles are expected to be tab-seperated files, with the first line being the header, 
      where the first column contains the true label and the second colum contains the predicted label.
"""
#!/usr/bin/env python3
import sys, getopt
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

def main(argv):
    inputfiles = ""
    titles = ""
    classes = ""
    is_norm = 0
    usage = "Usage: plot.py -i <inputfile> -t <title> -c <classes> -n <normalized>"

    try:
        opts, args = getopt.getopt(argv,"hi:t:c:n:",["ifiles=","titles=","classes=","norm="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-i", "--ifiles"):
            inputfiles = arg
        elif opt in ("-t", "--titles"):
            title = arg
        elif opt in ("-c", "--classes"):
            classes = arg
        elif opt in ("-n", "--norm"):
            is_norm = arg

    is_norm = True if is_norm == 1 else False
    titles = [t for t in title.split(";")]
    classes = [c for c in classes.split(";")]
    conf_matrices = [create_confusion_matrix(f) for f in inputfiles.split(";")]
    
    for cm, title in itertools.zip_longest(conf_matrices, titles):
        plt.figure()
        plot_confusion_matrix(cm, classes, title, is_norm)
        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])