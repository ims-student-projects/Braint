from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Convolution1D, MaxPooling1D, Flatten, Merge, Input

class LSTM_Model(object):
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
        self.lstm_dim = self.embedding_dim
        self.output_dim = output_dim
        self.weights = weights
        self.parse_params(params)
        self.model = self.create_LSTM()

    def parse_params(self, params=None):
        # TODO:
        # num_hidden_layers
        if 'attention' in params:
            self.attention = params['attention']
        else:
            self.attention = False
        if 'max_len' in params:
            self.max_sequence_len = params['max_len']
        else:
            self.max_sequence_len = None
        if 'dropout' in params:
            self.dropout = params['dropout']
        else:
            self.dropout = .5
        if 'trainable_embeddings' in params:
            self.trainable = params['trainable_embeddings']
        else:
            self.trainable = True
        if 'optimizer' in params:
            self.optimizer = params['optimizer']
        else:
            self.optimizer = 'adam'
        if 'loss' in params:
            self.loss = params['loss']
        else:
            self.loss = 'categorical_crossentropy'

    def create_LSTM(self):
        if self.attention:
            # BiLstm with attention
            # code from:
            # https://srome.github.io/Understanding-Attention-in-Neural-Networks-Mathematically/

            # Define the model
            inp = Input(shape=(self.max_sequence_len,))
            if self.weights:
                emb = Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        weights=[self.weights],
                        input_length=self.max_sequence_len,
                        trainable=True)(inp)
            else:
                emb = Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_length=self.max_sequence_len,
                        trainable=True)(inp)
            x = SpatialDropout1D(0.35)(emb)
            x = LSTM(self.lstm_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15)(x)
            x, attention = Attention()(x)
            x = Dense(6, activation="sigmoid")(x)

            model = Model(inputs=inp, outputs=x)
            model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=['accuracy'])

            attention_model = Model(inputs=inp, outputs=attention) # Model to print out the attention data

            return model, attention_model
        else:
            model = Sequential()
            if self.weights is not None:
                model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        weights=[self.weights],
                        input_length=self.max_sequence_len,
                        trainable=True))
            else:
                model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_length=self.max_sequence_len,
                        trainable=True))
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.lstm_dim))
            model.add(Dropout(self.dropout))
            model.add(Dense(self.output_dim, activation='softmax'))
    
            model.compile(optimizer=self.optimizer, loss=self.loss,
                  metrics=['accuracy'])
            return model

class BiLSTM_Model(object):

    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = self.embedding_dim
        self.output_dim = output_dim
        self.weights = weights
        self.parse_params(params)
        self.model = self.create_BiLSTM()

    def parse_params(self, params=None):
        # TODO:
        # num_hidden_layers
        if 'attention' in params:
            self.attention = params['attention']
        else:
            self.attention = False
        if 'max_len' in params:
            self.max_sequence_len = params['max_len']
        else:
            self.max_sequence_len = None
        if 'dropout' in params:
            self.dropout = params['dropout']
        else:
            self.dropout = .5
        if 'trainable_embeddings' in params:
            self.trainable = params['trainable_embeddings']
        else:
            self.trainable = True
        if 'optimizer' in params:
            self.optimizer = params['optimizer']
        else:
            self.optimizer = 'adam'
        if 'loss' in params:
            self.loss = params['loss']
        else:
            self.loss = 'categorical_crossentropy'


    def create_BiLSTM(self):
        if self.attention:
            # BiLstm with attention
            # code from:
            # https://srome.github.io/Understanding-Attention-in-Neural-Networks-Mathematically/

            # Define the model
            inp = Input(shape=(self.max_sequence_len,))
            if self.weights:
                emb = Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        weights=[self.weights],
                        input_length=self.max_sequence_len,
                        trainable=True)(inp)
            else:
                emb = Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_length=self.max_sequence_len,
                        trainable=True)(inp)
            x = SpatialDropout1D(0.35)(emb)
            x = Bidirectional(LSTM(self.lstm_dim, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
            x, attention = Attention()(x)
            x = Dense(6, activation="sigmoid")(x)

            model = Model(inputs=inp, outputs=x)
            model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=['accuracy'])

            attention_model = Model(inputs=inp, outputs=attention) # Model to print out the attention data

            return model, attention_model
        else:
            model = Sequential()
            if self.weights is not None:
                model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        weights=[self.weights],
                        input_length=self.max_sequence_len,
                        trainable=True))
            else:
                model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_length=self.max_sequence_len,
                        trainable=True))
            model.add(Dropout(self.dropout))
            model.add(Bidirectional(LSTM(self.lstm_dim)))
            model.add(Dropout(self.dropout))
            model.add(Dense(self.output_dim, activation='softmax'))
    
        
            model.compile(optimizer=self.optimizer, loss=self.loss,
                  metrics=['accuracy'])
            return model

class CNN_Model(object):

    def __init__(self, max_length, dim=300, dropout=.5, output_dim=8, weights=None):
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = weights
        self.max_length = max_length
        self.dim = embedding_dim
        self.dropout = dropout
        self.output_dim = output_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.trainable = trainable
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.model = self.create_cnn()

           
    def create_cnn(self):
    # Convolutional model
    graph_in = Input(shape=(self.max_length, len(self.embedding_dim)))
    convs = []
    for fsz in self.filter_sizes:
        conv = Convolution1D(nb_filter=self.num_filters,
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
    if self.weights is not None:
        model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        weights=[self.weights],
                        input_length=self.max_sequence_len,
                        trainable=True))
    else:
        model.add(Embedding(self.input_dim + 1,
                        self.embedding_dim,
                        input_length=self.max_sequence_len,
                        trainable=True))
    model.add(Dropout(self.dropout))
    model.add(graph)
    model.add(Dense(self.dim, activation='relu'))
    model.add(Dropout(self.dropout))
    model.add(Dense(self.output_dim, activation='softmax'))

    model.compile(optimizer=self.optimizer, loss=self.loss,
                  metrics=['accuracy'])
    return model
