"""
Models are partially based on code provided by Jeremy Barnes (https://github.com/jbarnesspain/sota_sentiment).
"""

from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, SpatialDropout1D, Concatenate

import sys

sys.path.append('../')

from utils.attention import Attention

class LSTM_Model(object):
    """ A simple LSTM architecture.

        The different parameters (params) are:
        max_len : int, the maximum sequence lenght
        dropout : int (between 0 and 1)
        trainable_embeddings: bool
        optimizer : the keras optimizer that should be used
        loss : the keras loss function that should be used
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        """ Inits the LSTM model.

            Args:
                vocab_size  :   int, size of the vocabulary
                embedding_dim:  int, dimensions of embeddings
                output_dim: int, number of classes
                weights :   (optional) pre-trained embeddings
                params  :   (optional) the model parameters to use if None default parameters are used
        """
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = self.embedding_dim
        self.output_dim = output_dim
        self.weights = weights
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
        self.model = self.__create_LSTM()

    def __create_LSTM(self):
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

class LSTM_ATT_Model(object):
        """ A LSTM architecture with attention mechanism.

        The different parameters (params) are:
        max_len : int, the maximum sequence lenght
        dropout : int (between 0 and 1)
        trainable_embeddings: bool
        optimizer : the keras optimizer that should be used
        loss : the keras loss function that should be used
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        """ Inits the LSTM+ATT model.

            Args:
                vocab_size  :   int, size of the vocabulary
                embedding_dim:  int, dimensions of embeddings
                output_dim: int, number of classes
                weights :   (optional) pre-trained embeddings
                params  :   (optional) the model parameters to use if None default parameters are used
        """
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = self.embedding_dim
        self.output_dim = output_dim
        self.weights = weights
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
        self.model = self.__create_LSTM_ATT()

    def __create_LSTM_ATT(self):
        inp = Input(shape=(self.max_sequence_len,))
        if self.weights is not None:
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
        x = SpatialDropout1D(self.dropout)(emb)
        x = LSTM(self.lstm_dim, return_sequences=True)(x)
        x, attention = Attention()(x)
        x = Dense(self.output_dim, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=['accuracy'])
        self.attention_model = Model(inputs=inp, outputs=attention)
        return model    

class BiLSTM_Model(object):
        """ A simple BiLSTM architecture.

        The different parameters (params) are:
        max_len : int, the maximum sequence lenght
        dropout : int (between 0 and 1)
        trainable_embeddings: bool
        optimizer : the keras optimizer that should be used
        loss : the keras loss function that should be used
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        """ Inits the BiLSTM model.

            Args:
                vocab_size  :   int, size of the vocabulary
                embedding_dim:  int, dimensions of embeddings
                output_dim: int, number of classes
                weights :   (optional) pre-trained embeddings
                params  :   (optional) the model parameters to use if None default parameters are used
        """
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = self.embedding_dim
        self.output_dim = output_dim
        self.weights = weights
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
        self.model = self.__create_BiLSTM()

    def __create_BiLSTM(self):
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

class BiLSTM_ATT_Model(object):
    """ A BiLSTM architecture with attention mechanism.

        The different parameters (params) are:
        max_len : int, the maximum sequence lenght
        dropout : int (between 0 and 1)
        trainable_embeddings: bool
        optimizer : the keras optimizer that should be used
        loss : the keras loss function that should be used
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        """ Inits the BiLSTM+ATT model.

            Args:
                vocab_size  :   int, size of the vocabulary
                embedding_dim:  int, dimensions of embeddings
                output_dim: int, number of classes
                weights :   (optional) pre-trained embeddings
                params  :   (optional) the model parameters to use if None default parameters are used
        """
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = self.embedding_dim
        self.output_dim = output_dim
        self.weights = weights
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
        self.model = self.__create_BiLSTM_ATT()

    def __create_BiLSTM_ATT(self):
        inp = Input(shape=(self.max_sequence_len,))
        if self.weights is not None:
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
        x = SpatialDropout1D(self.dropout)(emb)
        x = Bidirectional(LSTM(self.lstm_dim, return_sequences=True))(x)
        x, attention = Attention()(x)
        x = Dense(self.output_dim, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss=self.loss,
              optimizer=self.optimizer,
              metrics=['accuracy'])
        self.attention_model = Model(inputs=inp, outputs=attention)
        return model

class CNN_Model(object):
    """ A simple CNN architecture with one convolutional layer.

        The different parameters (params) are:
        max_len : int, the maximum sequence lenght
        dropout : int (between 0 and 1)
        trainable_embeddings: bool
        optimizer : the keras optimizer that should be used
        loss : the keras loss function that should be used
        filter_sizes : the filter sizes that should be used
        num_filters : the number of filters that should be used
    """
    def __init__(self, vocab_size, embedding_dim, output_dim, weights=None, params=None):
        """ Inits the CNN model.

            Args:
                vocab_size  :   int, size of the vocabulary
                embedding_dim:  int, dimensions of embeddings
                output_dim: int, number of classes
                weights :   (optional) pre-trained embeddings
                params  :   (optional) the model parameters to use if None default parameters are used
        """
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = weights
        self.dim = embedding_dim
        self.output_dim = output_dim
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
        if 'filter_sizes' in params:
            self.filter_sizes = params['filter_sizes']
        else:
            self.filter_sizes = (2,3,4)
        if 'num_filters' in params:
            self.num_filters = params['num_filters']
        else:
            self.num_filters = 3
        self.model = self.__create_cnn()
           
    def __create_cnn(self):
        # Convolutional model
        graph_in = Input(shape=(self.max_sequence_len, self.embedding_dim))
        convs = []
        for fsz in self.filter_sizes:
            conv = Conv1D(filters=self.num_filters,
                    kernel_size=fsz,
                    padding='valid',
                    activation='relu',
                    strides=1)(graph_in)
            pool = MaxPooling1D(pool_size=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)
    
        #out = Merge(mode='concat')(convs)
        out = Concatenate()(convs)
        graph = Model(inputs=graph_in, outputs=out)

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
