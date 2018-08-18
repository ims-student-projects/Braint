from corpus import Corpus
from model import Model
from evaluator.scorer import Scorer
from evaluator.result import Result


#test(self, path_model, path_weigths, classes_file, word_idx_file, max_len_file, test_corpus): 
def __evaluate_experiment(working_dir, save_dir, test_filename, test_labelsfile, path_weights, att):
    working_dir = working_dir
    test_file = working_dir + 'data/' + test_filename
    test_labels = working_dir + 'data/' + test_labelsfile
    #intialize corpus
    test_corpus = Corpus(test_file, test_labels)
    # create model object
    model = Model()
    result = model.test(save_dir, path_weights, test_corpus, att)
    scores = Result()
    s = Scorer(result)
    scores.show(0,s)

    print('Macro Fscore ')
    print(s.get_macro())

    print('Micro Fscore ')
    print(s.get_micro())

    for label in s.labels:
        print('Fscore ' + label)
        print(s.get_f_score(label))


def __run_experiment(working_dir, save_dir, train_filename, word_embed_filename, embed_file_type, architecture, max_len,
                        params=None, train_params=None, dev_filename=None, dev_labelsfile=None):
    # define path to working dir
    dir_path = working_dir
    # define path to dir where files should be saved to
    savedir_path = save_dir

    # get ful filenames paths to data files and initialize corpora
    train_file = dir_path + 'data/' + train_filename
    train_corpus = Corpus(train_file)
    if dev_filename and dev_labelsfile:
        dev_file = dir_path + 'data/' + dev_filename
        dev_labels = dir_path + 'data/' + dev_labelsfile
        dev_corpus = Corpus(dev_file, dev_labels)
    else:
        dev_corpus = None

    # define filename of embedding file and file type (txt or bin)
    word_embedding_file = dir_path + 'word_embeddings/' + word_embed_filename
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


def train_word_embedding_experiment(working_dir, train_filename, dev_file=None, dev_labels=None, test_file=None, test_labels=None):
    if not dev_file and not dev_labels:
        experiment_dir = working_dir + 'experiments/word_embedding_experiment/use_validation_split/'
    else:
        experiment_dir = working_dir + 'experiments/word_embedding_experiment/use_full_training_data/'
    architecture = 'LSTM'
    params = {'dropout' : .5, 'trainable_embeddings' : True}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    max_len = 60

    test = 'google.txt'
    # Word2Vec Embeddings
    # self trained
    word2vec_cbow_hs_twitter_300 = 'w2v_twitter_cbow_hs_300_w5.txt'
    word2vec_cbow_neg_twitter_300 = 'w2v_twitter_cbow_neg_300_w5.txt'
    word2vec_skip_hs_twitter_300 = 'w2v_twitter_skip_hs_300_w10.txt'
    word2vec_skip_neg_twitter_300 = 'w2v_twitter_skip_neg_300_w10.txt'
    # TODO
    #word2vec_cbow_neg_twitter_100 = ''
    #word2vec_skip_neg_twitter_100 = ''
    # from https://code.google.com/archive/p/word2vec/
    word2vec_neg_googlenews_300 = 'w2v_googlenews_neg_300.txt'
    
    # Glove Embeddings
    # add edone line to file with vocab size and dim so that they can be loaded like word2vec
    # from https://nlp.stanford.edu/projects/glove/
    #glove_wikipedia+gigaword_300 = ''
    #glove_wikipedia+gigaword_100 = ''
    glove_twitter_200 = 'converted_glove.twitter.27B.200d.txt'
    glove_twitter_100 = 'converted_glove.twitter.27B.100d.txt'

    # Experiments
    print('Running Word Embedding Experiments')
    print('Using full training data')
    # print('Test Experiment- Word embeddings: google')
    # __run_experiment(working_dir, experiment_dir + 'google_test/', train_filename, test, 'word2vec', 
    #                 architecture, params, train_params, dev_file, dev_labels)

    print('Experiment 1/7 - Word embeddings: word2vec_cbow_hs_twitter_300')
    __run_experiment(working_dir, experiment_dir + 'word2vec_cbow_hs_twitter_300/', train_filename, word2vec_cbow_hs_twitter_300, 'word2vec', 
                    architecture, max_len, params, train_params, dev_file, dev_labels)

    print('Experiment 2/7 - Word embeddings: word2vec_cbow_neg_twitter_300')
    __run_experiment(working_dir, experiment_dir + 'word2vec_cbow_neg_twitter_300/', train_filename, word2vec_cbow_neg_twitter_300, 'word2vec', 
                    architecture, max_len, params, train_params, dev_file, dev_labels)

    print('Experiment 3/7 - Word embeddings: word2vec_skip_hs_twitter_300')
    __run_experiment(working_dir, experiment_dir + 'word2vec_skip_hs_twitter_300/', train_filename, word2vec_skip_hs_twitter_300, 'word2vec', 
                    architecture, max_len, params, train_params, dev_file, dev_labels)

    print('Experiment 4/7 - Word embeddings: word2vec_skip_neg_twitter_300')
    __run_experiment(working_dir, experiment_dir + 'word2vec_skip_neg_twitter_300/', train_filename, word2vec_skip_neg_twitter_300, 'word2vec', 
                    architecture, max_len, params, train_params, dev_file, dev_labels)

    print('Experiment 5/7 - Word embeddings: word2vec_neg_googlenews_300')
    __run_experiment(working_dir, experiment_dir + 'word2vec_neg_googlenews_300/', train_filename, word2vec_neg_googlenews_300, 'word2vec', 
                   architecture, max_len, params, train_params, dev_file, dev_labels)

    print('Experiment 6/7 - Word embeddings: glove.twitter.27B.100')
    __run_experiment(working_dir, experiment_dir + 'glove.twitter.27B.100/', train_filename, glove_twitter_100, 'word2vec', 
                    architecture, max_len, params, train_params, dev_file, dev_labels)

    print('Experiment 7/7 - Word embeddings: glove.twitter.27B.200')
    __run_experiment(working_dir, experiment_dir + 'glove.twitter.27B.200/', train_filename, glove_twitter_200, 'word2vec', 
                   architecture, max_len, params, train_params, dev_file, dev_labels)


    # TODO
    # Evaluate model
    # if test_file and test.labels

def eval_word_embedding_experiment(working_dir, test_filename, test_labelsfile):
    experiment_dir = working_dir + 'experiments/word_embedding_experiment/use_full_training_data/'

    word2vec_cbow_hs_twitter_300_weigths = experiment_dir + 'word2vec_cbow_hs_twitter_300/' + 'weights-improvement-08-0.63.hdf5'
    word2vec_cbow_neg_twitter_300_weigths = experiment_dir + 'word2vec_cbow_neg_twitter_300/' + 'weights-improvement-04-0.63.hdf5'
    word2vec_skip_hs_twitter_300_weigths = experiment_dir + 'word2vec_skip_hs_twitter_300/' + 'weights-improvement-03-0.62.hdf5'
    word2vec_skip_neg_twitter_300_weights = experiment_dir + 'word2vec_skip_neg_twitter_300/' + 'weights-improvement-04-0.63.hdf5'
    glove_twitter_100_weights = experiment_dir + 'glove.twitter.27B.100/' + 'weights-improvement-05-0.61.hdf5'
    #glove.twitter.27B.200_weights = experiment_dir + 'glove.twitter.27B.200/' + 

    print('Wordembedding Experiment Evaluation')
    print('Experiment 1/7 - Word embeddings: word2vec_cbow_hs_twitter_300')
    __evaluate_experiment(working_dir, experiment_dir + 'word2vec_cbow_hs_twitter_300/', test_filename, test_labelsfile, word2vec_cbow_hs_twitter_300_weigths)
    
    print('Experiment 2/7 - Word embeddings: word2vec_cbow_neg_twitter_300')
    __evaluate_experiment(working_dir, experiment_dir + 'word2vec_cbow_neg_twitter_300/', test_filename, test_labelsfile, word2vec_cbow_neg_twitter_300_weigths)

    print('Experiment 3/7 - Word embeddings: word2vec_skip_hs_twitter_300')
    __evaluate_experiment(working_dir, experiment_dir + 'word2vec_skip_hs_twitter_300/', test_filename, test_labelsfile, word2vec_skip_hs_twitter_300_weigths)

    print('Experiment 4/7 - Word embeddings: word2vec_skip_neg_twitter_300')
    __evaluate_experiment(working_dir, experiment_dir + 'word2vec_skip_neg_twitter_300/', test_filename, test_labelsfile, word2vec_skip_neg_twitter_300_weights)

    print('Experiment 5/7 - Word embeddings: glove.twitter.27B.100')
    __evaluate_experiment(working_dir, experiment_dir + 'glove.twitter.27B.100/', test_filename, test_labelsfile, glove_twitter_100_weights)

    #print('Experiment 6/7 - Word embeddings: glove.twitter.27B.200')
    #__evaluate_experiment(working_dir, glove.twitter.27B.200_weights + 'eval/', test_filename, test_labelsfile, glove.twitter.27B.200_weights)



    ####TODO experiment without trainable embeddings!!!!


def eval_architectures_experiment(working_dir, test_filename, test_labelsfile):
    experiment_dir = working_dir + 'experiments/architecture_experiment/use_full_training_data/'

    lstm_att_weights = experiment_dir + 'LSTM+ATT/' + 'weights-improvement-02-0.64.hdf5'
    bilstm_weights =  experiment_dir + 'BiLSTM/' + 'weights-improvement-06-0.64.hdf5'
    bilstm_att_weights =  experiment_dir + 'BiLSTM+ATT/' + 'weights-improvement-03-0.64.hdf5'
    cnn_weights =  experiment_dir + 'CNN/' + 'weights-improvement-07-0.55.hdf5'

    print('Architecture Experiment Evaluation')
    print('trained using word embedding: ' + 'w2v_twitter_cbow_hs_300_w5.txt')

    print('Experiment 1/4 - Architectures: LSTM + ATT')
    __evaluate_experiment(working_dir, experiment_dir + 'LSTM+ATT/', test_filename, test_labelsfile, lstm_att_weights, att=True)

    assert(False)

    print('Experiment 2/4 - Architectures: BiLSTM')
    __evaluate_experiment(working_dir, experiment_dir + 'BiLSTM/', test_filename, test_labelsfile, bilstm_weights)

    print('Experiment 3/4 - Architectures: BiLSTM + ATT')
    __evaluate_experiment(working_dir, experiment_dir + 'BiLSTM+ATT/', test_filename, test_labelsfile, bilstm_att_weights)

    print('Experiment 4/4 - Architectures: CNN')
    __evaluate_experiment(working_dir, experiment_dir + 'CNN/', test_filename, test_labelsfile, cnn_weights)


def train_architecture_experiment(working_dir, train_filename, dev_file=None, dev_labels=None, test_file=None, test_labels=None):
    experiment_dir = 'experiments/architecture_experiment/use_validation_split'
    if not dev_file and not dev_labels:
        experiment_dir = working_dir + 'experiments/architecture_experiment/use_validation_split/'
    else:
        experiment_dir = working_dir + 'experiments/architecture_experiment/use_full_training_data/'

    # embeddings
    word2vec_cbow_hs_twitter_300 = 'w2v_twitter_cbow_hs_300_w5.txt'
    max_len = 60


    # Experiments
    print('Running Architectures Experiments')
    print('Using full training data')


    architecture = 'CNN'
    params = {'dropout' : .5, 'trainable_embeddings' : True}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    print('1/4 Experiment Architectures: CNN')
    __run_experiment(working_dir, experiment_dir + 'CNN/', train_filename, word2vec_cbow_hs_twitter_300, 'word2vec', 
                      architecture, max_len, params, train_params, dev_file, dev_labels)

    
    architecture = 'BiLSTM'
    params = {'dropout' : .5, 'trainable_embeddings' : True}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    print('2/4 Test Experiment- Architectures: BiLSTM')
    __run_experiment(working_dir, experiment_dir + 'BiLSTM/', train_filename, word2vec_cbow_hs_twitter_300, 'word2vec', 
                      architecture, max_len, params, train_params, dev_file, dev_labels)


    architecture = 'BiLSTM'
    params = {'dropout' : .5, 'trainable_embeddings' : True, 'attention' : True}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    print('3/4 Test Experiment- Architectures: BiLSTM with attention')
    __run_experiment(working_dir, experiment_dir + 'BiLSTM+ATT/', train_filename, word2vec_cbow_hs_twitter_300, 'word2vec', 
                     architecture, max_len, params, train_params, dev_file, dev_labels)


    architecture = 'LSTM'
    params = {'dropout' : .5, 'trainable_embeddings' : True, 'attention' : True}
    train_params = {'num_epochs' : 15, 'min_count' : 1}

    print('4/4 Test Experiment- Architectures: LSTM with attention')
    __run_experiment(working_dir, experiment_dir + 'LSTM+ATT/', train_filename, word2vec_cbow_hs_twitter_300, 'word2vec', 
                     architecture, max_len, params, train_params, dev_file, dev_labels)



def parameter_tuning_experiment():
    experiment_dir = 'experiments/parameter_tuning_experiment/use_validation_split'
    pass


def wassa_submission():
    working_dir = '/Users/marina/Documents/Master/2018_SS/TeamLab/'
    experiment_dir = working_dir + 'experiments/architecture_experiment/use_full_training_data/'

    save_dir = working_dir + 'wassa_submission_blstm+ATT/'

    test_file = working_dir + 'data/test.csv'

    bilstm_att_weights =  experiment_dir + 'BiLSTM+ATT/' + 'weights-improvement-03-0.64.hdf5'

    #intialize corpus
    test_corpus = Corpus(test_file, None, False, False)
    # create model object
    model = Model()
    model.test(experiment_dir + 'BiLSTM+ATT/', bilstm_att_weights, test_corpus, True)



def main():
    working_dir = '/Users/marina/Documents/Master/2018_SS/TeamLab/'
    train_filename = 'train-v2.csv'
    #dev_file = None
    #dev_labels = None
    dev_file = 'trial-v2.csv'
    dev_labels = 'trial-v2.labels'
    #test_file = 'trial.csv'
    #test_labels = 'trial.labels'
    #test_file = None
    #test_labels = None

    #train_word_embedding_experiment(working_dir, train_filename, dev_file, dev_labels, test_file, test_labels)
    #train_architecture_experiment(working_dir, train_filename, dev_file, dev_labels, test_file, test_labels)
    #eval_word_embedding_experiment(working_dir, dev_file, dev_labels)
    eval_architectures_experiment(working_dir, dev_file, dev_labels)

if __name__ == "__main__":
    main()