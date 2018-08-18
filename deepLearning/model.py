import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

import sys

sys.path.append('../')

from utils.WordVecs import *
from architectures import LSTM_Model , BiLSTM_Model, CNN_Model
from corpus import Corpus
from tokenizer import Tokenizer

class Model(object):

    def __init__(self):
        self.__tokenizer = Tokenizer()

    def __tweet2idx(self, tweet, w2idx):
        # maps tokens to ids
        return np.array([w2idx[token] if token in w2idx else w2idx['<UNK>'] for token in tweet])

    def __convert_format(self, corpus, classes, w2idx, max_len, only_predict:bool=False):
        dataset = []
        for tweet in corpus:
            dataset.append((tweet.get_text(), tweet.get_gold_label()))
        x_data, y_data = zip(*dataset)
        x_data = [self.__tokenizer.get_only_tokens(tweet) for tweet in x_data]
        # make it possible to get predictions for unlabeled tweets
        if not only_predict:
            y_data = [classes[label] for label in y_data]
            # class to one hot vector
            y_data = [np.eye(len(classes))[label] for label in y_data]
            y_data = np.array(y_data)
        else:
            y_data = None
        # to np array
        x_data = np.array([self.__tweet2idx(tweet, w2idx) for tweet in x_data])     
        # padding
        x_data = pad_sequences(x_data, max_len)
        return x_data, y_data

    def __create_vocab(self, corpus):
        vocab = {}
        for tweet in corpus:
            for token in self.__tokenizer.get_only_tokens(tweet.get_text()):
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        # add <unk> token to map unseen words to, use high nuber so that it does not get filtered out by min_count
        vocab['<UNK>'] = 100
        return vocab

    def __get_word_embeddings(self, vecs, vocab, min_count):
        dim = vecs.vector_size
        embeddings = {}
        for word in vecs._w2idx.keys():
            embeddings[word] = vecs[word]
        # add random embeddings for words that occur in training data but not in the pretrained w2v embeddings
        for word in vocab:
            if word not in embeddings and vocab[word] >= min_count:
                embeddings[word] = np.random.uniform(-0.25, 0.25, dim)

        vocab_size = len(embeddings)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, dim), dtype='float32')
        W[0] = np.zeros(dim, dtype='float32')
        i = 1
        for word in embeddings:
            W[i] = embeddings[word]
            word_idx_map[word] = i
            i += 1  
        return embeddings, W, word_idx_map    
    
    def train(self, train_corpus, classes, architecture, params, num_epochs:int, max_len:int, embedding_file, file_type, min_count:int, save_dir, dev_corpus=None):
        """ Function to train a model.

            Args:
                train_corpus : Corpus containg the Tweets for training 
                classes      : Dictionary containing a mapping from classification classes to ids
                architecture : String one of LSTM, BiLSTM, CNN, LSTM+ATT, BiLSTM+ATT
                params       : Dictionary containing a mapping from model parameters to values
                num_epochs   : int, number of training iterations
                max_len      : int, maximum sequence length
                embedding_file : String, name/path to pretrained word embedding file
                file_type    : String, Word2Vec (for text) or binary
                min_count    : int, minum number of occurences
                save_dir     : String, directiory to save model and weights to
                dev_corpus   :(optional) Corpus containing Tweets for development if None a validation split of 90/10 is used

            Files that are written:
            'attention_model.h5' : only if architecture is LSTM+ATT or BiLSTM+ATT
            'model.json'         : model architecture
            'vocab.p'            : vocab of traing data
            'max_sequence_len.p' : maximum sequence length
            'word_idx_map.p'     : word to id mapping
            'classes.p'          : classes to id mapping

        """
        # restrict gpu memory consumption
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
        params['max_len'] = max_len
        print('Creating vocab...')
        vocab = self.__create_vocab(train_corpus)
        print('vocab finished')
        # create wordvecs and W
        print('Loading embeddings...')
        # filter embedding file with vocab
        vecs = WordVecs(embedding_file, file_type, vocab)
        print('finished loading')
        print('Creating wordvecs, W and w2idx map...')
        embeddings, W, word_idx_map = self.__get_word_embeddings(vecs, vocab, min_count)
        print('wordvecs, W, w2idx map finished')
        # convert train corpus to xtrain, ytrain
        print('Converting train corpus...')
        x_train, y_train = self.__convert_format(train_corpus, classes, word_idx_map, max_len)
        # convert dev corpus to xdev, ydev
        if dev_corpus:
            print('Converting dev corpus...')
            x_dev, y_dev = self.__convert_format(dev_corpus, classes, word_idx_map, max_len)
        print('converting finished')
        # create nn
        output_dim = len(classes)
        vocab_size = len(embeddings)
        embedding_dim = vecs.vector_size
        print('Creating nn...')
        if architecture == 'LSTM':
            nn = LSTM_Model(vocab_size, embedding_dim, output_dim, W, params)
        if architecture == 'LSTM+ATT':
            nn = LSTM_Model(vocab_size, embedding_dim, output_dim, W, params)
        elif architecture == 'BiLSTM':
            nn = BiLSTM_Model(vocab_size, embedding_dim, output_dim, W, params)
        elif architecture == 'BiLSTM+ATT':
            nn = BiLSTM_Model(vocab_size, embedding_dim, output_dim, W, params)
        elif architecture == 'CNN':
            nn = CNN_Model(vocab_size, embedding_dim, output_dim, W, params)
        else:
            return
        print('nn finished')      
        #checkpointing
        filepath = save_dir + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        # train
        if dev_corpus:
            nn.model.fit(x_train, y_train, validation_data=[x_dev, y_dev], epochs=num_epochs, verbose=1, callbacks=callbacks)
        else:
            nn.model.fit(x_train, y_train, validation_split=0.1, epochs=num_epochs, verbose=1, callbacks=callbacks)
        print('Finished training ' + architecture)   
        # serialize attention model
        if architecture == 'LSTM+ATT' or architecture == 'BiLSTM+ATT':
            attention_model = nn.attention_model
            attention_model.save(save_dir + 'attention_model.h5')
        # serialize model architecture to JSON
        model_json = nn.model.to_json()
        with open(save_dir + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize vocab, word to id mapping, max_len and classes
        pickle.dump(vocab, open(save_dir + "vocab.p", "wb"))
        pickle.dump(max_len, open(save_dir + "max_sequence_len.p", "wb"))
        pickle.dump(word_idx_map, open(save_dir + "word_idx_map.p", "wb"))
        pickle.dump(classes, open(save_dir + "classes.p", "wb" ))

    def get_word_attention(self, saved_dir, test_corpus):
        """ Gets attention score for all tokens in the tweet for every Tweet in the test_corpus. 

            Args:
                saved_dir    : String, path/name to file where weights of the attention model, 
                                        the file containg the max_sequence_length (for padding; assumed to be called "max_sequence_len.p") and 
                                        token to id mapping (assumed to be called word_idx_map.p) are stored
                test_corpus  : Corpus, containg the Tweets for which attentions should be calculated
            
            Returns:
                list of list of tupels containg the attention values for each token.
        """
        inv_classes = {v: k for k, v in classes.items()}
        max_len = pickle.load(open(save_dir + "max_sequence_len.p", "rb"))
        word_idx_map = pickle.load(open(save_dir + "word_idx_map.p", "rb"))
        inv_word_idx_map = {v: k for k, v in word_idx_map.items()}
        # convert test data into input format
        x_test, y_test = self.__convert_format(test_corpus, classes, word_idx_map, max_len)
        # load model
        attention_model = load_model(saved_dir)
        print("Attention model loaded from disk")
        # compile model
        attention_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # get attentions
        word_attentions = []
        attentions = attention_model.predict(x_test, verbose=1)
        for importances, tweet in zip(attentions, x_test):
            temp = []
            for importance, token in zip(importances, tweet):
                if token != 0:
                    temp.append((inv_word_idx_map[token], importance))
            word_attentions.append(temp)
        return word_attentions

    def predict(self, saved_dir, path_weights, test_corpus):
        """ Gets predictions tweet for every Tweet in the test_corpus and writes them into the Tweet data structure as well as to file. 

            Args:
                saved_dir    : String, path/name to file where weights of the model,
                                        architecture of the model (assumed to be called model.json)
                                        the file containg the max_sequence_length (for padding; assumed to be called "max_sequence_len.p"), 
                                        token to id mapping (assumed to be called word_idx_map.p), 
                                        classes to id mapping (assumed to be called classes.p)
                                        are stored
                path_weights : String, path/name to file where model weights are stored
                test_corpus  : Corpus, containg the Tweets for which attentions should be calculated
            
            Writes file
            'predictions.csv' containg the predition of the model as well as the tweet itself
            into the saved_dir.

        """
        classes = pickle.load(open(save_dir + "classes.p", "rb"))
        inv_classes = {v: k for k, v in classes.items()}
        max_len = pickle.load(open(save_dir + "max_sequence_len.p", "rb"))
        word_idx_map = pickle.load(open(save_dir + "word_idx_map.p", "rb"))
        inv_word_idx_map = {v: k for k, v in word_idx_map.items()}
        # convert test data into input format
        x_test, _ = self.__convert_format(test_corpus, classes, word_idx_map, max_len, True)
        # load model architecture
        json_file = open(saved_dir + 'model.json', 'r')
        loaded_model = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model)
        # load weights
        model.load_weights(path_weights)
        print("Loaded model from disk")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # get predictions
        predictions = model.predict(x_test, verbose=1)
        # one hot vector to class
        predictions = [np.argmax(prediction,axis=-1) for prediction in predictions]
        true_labels = [np.argmax(label,axis=-1) for label in y_test]
        # write predictions to file 
        i = 0
        with open(saved_dir + 'predictions.csv', 'w') as out: 
            out.write("tpredicted_label\ttweet\n")
            for prediction in predictions:
                pred_label = inv_classes[prediction]
                text = test_corpus.get_ith(i).get_text()
                out.write(pred_label + "\t" + text + "\n")
                i += 1          
        # write predictions in tweets of test corpus
        # order of predictions should be the same as oder of tweets in test_corpus
        for i in range(len(predictions)):
            test_corpus.get_ith(i).set_pred_label(inv_classes[predictions[i]])
        return test_corpus

    def test(self, save_dir, path_weights, test_corpus):
        """ Tests predictions of the models for the Tweets in test_corpus. Writes predictions into the Tweet data structure as well as to file. 

            Args:
                saved_dir    : String, path/name to file where weights of the model,
                                        architecture of the model (assumed to be called model.json)
                                        the file containg the max_sequence_length (for padding; assumed to be called "max_sequence_len.p"), 
                                        token to id mapping (assumed to be called word_idx_map.p), 
                                        classes to id mapping (assumed to be called classes.p)
                                        are stored
                path_weights : String, path/name to file where model weights are stored
                test_corpus  : Corpus, containg the Tweets for which attentions should be calculated
            
            Writes file
            'predictions.csv' containg the predition of the model as well as the tweet itself
            into the saved_dir.
        """ 
        classes = pickle.load(open(save_dir + "classes.p", "rb"))
        inv_classes = {v: k for k, v in classes.items()}
        max_len = pickle.load(open(save_dir + "max_sequence_len.p", "rb"))
        word_idx_map = pickle.load(open(save_dir + "word_idx_map.p", "rb"))
        inv_word_idx_map = {v: k for k, v in word_idx_map.items()}
        # convert test data into input format
        x_test, y_test = self.__convert_format(test_corpus, classes, word_idx_map, max_len)
        # load model architecture
        json_file = open(save_dir + 'model.json', 'r')
        loaded_model = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model)
        # load weights
        model.load_weights(path_weights)
        print("Loaded model from disk")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # evaluate model & print accuracy
        score = model.evaluate(x_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        # get predictions
        predictions = model.predict(x_test, verbose=1)
        # one hot vector to class
        predictions = [np.argmax(prediction,axis=-1) for prediction in predictions]
        true_labels = [np.argmax(label,axis=-1) for label in y_test]
        # write predictions to file 
        i = 0
        with open(save_dir + 'predictions.csv', 'w') as out:
            out.write("true_label\tpredicted_label\ttweet\n")
            for label, prediction in zip(true_labels, predictions):
                pred_label = inv_classes[prediction]
                true_label = inv_classes[label]
                text = test_corpus.get_ith(i).get_text()
                out.write(true_label + "\t" + pred_label + "\t" + text + "\n")
                i += 1          
        # write predictions in tweets of test corpus
        # order of predictions should be the same as oder of tweets in test_corpus
        for i in range(len(predictions)):
            test_corpus.get_ith(i).set_pred_label(inv_classes[predictions[i]])
        return test_corpus
