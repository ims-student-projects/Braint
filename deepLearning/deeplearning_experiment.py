import os
import sys

sys.path.append('../')

from corpus import Corpus
from model import Model
from evaluator.scorer import Scorer

# maximum sequence length
max_len = 60


#test(self, path_model, path_weigths, classes_file, word_idx_file, max_len_file, test_corpus): 
def __evaluate_experiment(working_dir, save_dir, test_filename, test_labelsfile, path_weights):
    working_dir = working_dir
    test_file = working_dir + test_filename
    test_labels = working_dir + test_labelsfile
    #intialize corpus
    test_corpus = Corpus(test_file, test_labels)
    # create model object
    model = Model()
    result = model.test(save_dir, path_weights, test_corpus)
    scorer = Scorer(result)

    print('Macro Fscore:')
    print(scorer.get_macro())

    print('Micro Fscore:')
    print(scorer.get_micro())
    for label in scorer.labels:
        print('Fscore ' + label)
        print(scorer.get_f_score(label))


def __run_experiment(working_dir, save_dir, train_filename, word_embed_filename, embed_file_type, architecture, max_len,
                        params=None, train_params=None, dev_filename=None, dev_labelsfile=None):
    # define path to working dir
    dir_path = working_dir
    # define path to dir where files should be saved to
    savedir_path = save_dir

    # get ful filenames paths to data files and initialize corpora
    train_file = dir_path + train_filename
    train_corpus = Corpus(train_file)
    if dev_filename and dev_labelsfile:
        dev_file = dir_path + dev_filename
        dev_labels = dir_path + dev_labelsfile
        dev_corpus = Corpus(dev_file, dev_labels)
    else:
        dev_corpus = None

    # define filename of embedding file and file type (txt or bin)
    word_embedding_file = dir_path + 'word_embeddings/w2v_embeddings/' + word_embed_filename
    file_type = embed_file_type

    # define label to int mapping
    classes = {'joy' : 0, 'anger' : 1, 'fear' : 2, 'surprise' : 3, 'disgust' : 4, 'sad' : 5}

    architecture = architecture
    params = params

    # parse train parameters
    if 'num_epochs' in train_params:
        num_epochs = train_params['num_epochs']
    else:
        num_epochs = 5
    if 'min_count' in train_params:
        min_count = train_params['min_count']
    else:
        min_count = 1

    # create model object
    model = Model()
    # train model TODO: use savedir
    model.train(train_corpus, classes, architecture, params, num_epochs, max_len, word_embedding_file, file_type, min_count, save_dir, dev_corpus)
    # free memory
    del model

def train_architecture_experiment(working_dir, experiment_dir, word_embeddings, architecture, train_filename, dev_file, dev_labels, test_file, test_labels):

    # Experiments
    print('Running Architectures Experiment')
    print('Using full training data')

    params = {'dropout' : .5, 'trainable_embeddings' : True}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    # create output directory
    directory = os.path.dirname(working_dir + experiment_dir + architecture + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Experiment Architecture: ' + architecture)
    __run_experiment(working_dir, working_dir + experiment_dir + architecture + '/', train_filename, word_embeddings, 'word2vec', 
                      architecture, max_len, params, train_params, dev_file, dev_labels)

def train_word_embedding_experiment(working_dir, experiment_dir, word_embeddings, architecture, train_filename, dev_file, dev_labels, test_file, test_labels):

    # Experiments
    print('Running Word Embedding Experiments')
    print('Using full training data')

    params = {'dropout' : .5, 'trainable_embeddings' : False}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    # create output directory
    directory = os.path.dirname(working_dir + experiment_dir + word_embeddings + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('Experiment Word embeddings: ' + word_embeddings)
    __run_experiment(working_dir, working_dir + experiment_dir + word_embeddings + '/', train_filename, word_embeddings, 'word2vec', 
                   architecture, max_len, params, train_params, dev_file, dev_labels)

def eval_architecture_experiment(working_dir, experiment_dir, architecture, best_weights, test_filename, test_labelsfile):

    architecture_path = working_dir + experiment_dir + architecture + '/'
    print(architecture_path)
    print('Architecture Experiment Evaluation')

    print('Experiment Architecture: ' + architecture)
    __evaluate_experiment(working_dir, architecture_path, test_filename, test_labelsfile, architecture_path + best_weights)

def eval_word_embedding_experiment(working_dir, experiment_dir, word_embeddings, best_weights, test_filename, test_labelsfile):

    word_embedding_path = working_dir + experiment_dir + word_embeddings + '/'
    
    print('Wordembedding Experiment Evaluation')

    print('Experiment Word embeddings: ' + word_embeddings)
    __evaluate_experiment(working_dir, word_embedding_path, test_filename, test_labelsfile, word_embedding_path + best_weights)


def main():
    # absolute path
    working_dir = '/run/media/martin/Elements/Marina/TeamLab/'
    # relative to working_dir
    experiment_dir = 'experiments/architecture_experiment/'
    # directory of train, dev and test data, relative to working dir
    data_dir = 'data/'
    # filename of word_embeddings
    word_embeddings = 'w2v_cbow_hs_300_w5.txt'
    # architecture
    architecture = 'BiLSTM'
    # filename of best weights
    best_weights = 'weights-improvement-07-0.64.hdf5'
    
    # filenames of the train, dev and test data
    train_filename = 'train-v2.csv'
    dev_file = 'trial-v2.csv'
    dev_labels = 'trial-v2.labels'
    test_file = 'trial.csv'
    test_labels = 'trial.labels'

    # combine filenames for internal use
    train_f = data_dir + train_filename
    dev_f = data_dir + dev_file
    dev_l = data_dir + dev_labels
    test_f = data_dir + test_file
    test_l = data_dir + test_labels

    # train_word_embedding_experiment(working_dir, experiment_dir, word_embeddings, architecture, train_f, dev_f, dev_l, test_f, test_l)
    train_architecture_experiment(working_dir, experiment_dir, word_embeddings, architecture, train_f, dev_f, dev_l, test_f, test_l)
    # eval_word_embedding_experiment(working_dir, experiment_dir, word_embeddings, best_weights, test_f, test_l)
    # eval_architecture_experiment(working_dir, experiment_dir, architecture, best_weights, test_f, test_l)

if __name__ == "__main__":
    main()
