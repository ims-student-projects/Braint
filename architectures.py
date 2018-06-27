from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Embedding, Convolution1D, MaxPooling1D, Flatten, Merge, Input

class LSTM(object):
    """
        Create simple one layer lstm
        lstm_dim: dimension of hidden layer
        dropout: 0-1
        weights: if you have pretrained embeddings, you can include them here
        train: if true, updates the original word embeddings
    """

    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.weights = weights
        self.parse_params(params)
        self.model = create_LSTM()

    def parse_params(self, params=None):
        # TODO:
        # num_hidden_layers
        # + attention
        # optimizer
        # loss
        if 'max_len' in params:
            self.max_sequence_len = params['max_len']
        else:
            self.max_sequence_len = None
        if 'lstm_dim' in params:
            self.lstm_dim = params['lstm_dim']
        else:
            self.lstm_dim = 300
        if 'droput' in params:
            self.dropout = params['dropout']
        else:
            self.droput = .5
        if 'train' in params:
            self.trainable = params['train']
        else:
            self.trainable = True

    def create_LSTM(self):
        model = Sequential()
        if self.weights:
            model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_len=self.max_sequence_len,
                        weights=[self.weights],
                        trainable=self.trainable))
        else:
            model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_len=self.max_sequence_len
                        trainable=self.trainable))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.lstm_dim))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.output_dim, activation='softmax'))
    
        if output_dim == 2:
            model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
        else:
            model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class BiLSTM(object):

    def __init__(vocab_size, embedding_dim, output_dim, weights=None, params=None):
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.weights = weights
        self.parse_params(params)
        self.model = create_LSTM()

    def parse_params(self, params=None):
        # TODO:
        # num_hidden_layers
        # + attention
        # optimizer
        # loss
        if 'max_len' in params:
            self.max_sequence_len = params['max_len']
        else:
            self.max_sequence_len = None
        if 'lstm_dim' in params:
            self.lstm_dim = params['lstm_dim']
        else:
            self.lstm_dim = 300
        if 'droput' in params:
            self.dropout = params['dropout']
        else:
            self.droput = .5
        if 'train' in params:
            self.trainable = params['train']
        else:
            self.trainable = True


    def create_BiLSTM(self):
        model = Sequential()
        model = Sequential()
        if self.weights:
            model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_len=self.max_sequence_len,
                        weights=[self.weights],
                        trainable=self.trainable))
        else:
            model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_len=self.max_sequence_len
                        trainable=self.trainable))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(self.lstm_dim)))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.output_dim, activation='softmax'))
    
        if output_dim == 2:
            model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
        else:
            model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class CNN(object):

    def __init__(self, W, max_length, dim=300, dropout=.5, output_dim=8):
        self.W = W
        self.max_length = max_length
        self.dim = dim
        self.dropout = dropout
        self.output_dim = output_dim
        self.model = create_cnn()
            
    def create_cnn(self):

    # Convolutional model
    filter_sizes=(2,3,4)
    num_filters = 3
   
    graph_in = Input(shape=(self.max_length, len(self.W[0])))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                 filter_length=fsz,
                 border_mode='valid',
                 activation='relu',
                 subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
        
    out = Merge(mode='concat')(convs)
    graph = Model(input=graph_in, output=out)

    # Full model
    model = Sequential()
    model.add(Embedding(output_dim=self.W.shape[1],
                        input_dim=self.W.shape[0],
                        input_length=self.max_length, weights=[self.W],
                        trainable=True))
    model.add(Dropout(self.dropout))
    model.add(graph)
    model.add(Dense(self.dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(self.output_dim, activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

