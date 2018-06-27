import pickle
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from Utils.WordVecs import *

from evaluator.scorer import Scorer
from models import LSTM, BiLSTM, CNN
from corpus import Corpus
from tokenizer import Tokenizer


tokenizer = Tokenizer()
classes = {'sad': 0, 'joy': 1}

def tweet2idx(tweet, w2idx):
    if not 
    return np.array([w2idx[w] if w in w2idx else [w2idx['<UNK>']] for token in tweet])

def convert_format(corpus, classes, w2idx, max_len):
    dataset = []
    for tweet in corpus:
        dataset.append((tweet.get_gold_label(), tweet.get_text()))
    x_data, y_data = zip(*dataset)
    x_data = [tokenizer.get_tokens(tweet) for tweet in x_data]
    y_data = [classes[label] for label in y_data]
    # class to one hot vector
    y_data = [np.eye(len(classes))[label] for label in y_data]
    # to np array
    x_data = np.array([tweet2idx(tweet, w2idx) for tweet in x_data])
    y_data = np.array(y_data)
    # padding
    x_data = pad_sequences(x_data, max_len)
    return x_data, y_data

def create_vocab(train_corpus):
    vocab = {}
    for tweet in corpus:
        for token in tokenizer.get_tokens(tweet):
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    # add <unk> token to map unseen words to, use high nuber so that it does not get filtered out by min_count
    vocab['<UNK>'] = 100
    return vocab

def get_word_embeddings(vecs, vocab, min_count):
    # vecs : self.vocab_length, self.vector_size, self._matrix, self._w2idx, self._idx2w
    dim = vecs.vector_size
    embeddings = {}
    for word in vecs._w2idx.keys():
        embeddings[word] = vecs[word]
    # add random embeddings for words that occur in training data but not in the pretrained embeddings
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
    
def train(train_corpus, dev_corpus, classes, architecture='LSTM', params, num_iter, max_input_len, embedding_file, file_type='bin', min_count):
    vocab = create_vocab(train_corpus)
    # create wordvecs and W
    vecs = WordVecs(embedding_file, file_type)
    embeddings, W, word_idx_map = get_word_embeddings(vecs, vocab, min_count)
    # convert train corpus to xtrain, ytrain
    x_train, y_train = convert_format(train_corpus, classes, word_idx_map, max_input_len)
    # convert dev corpus to xdev, ydev
    x_dev, y_dev = convert_format(dev_corpus, classes, word_idx_map, max_len)
    # create model
    output_dim = len(classes)
    if architecture == 'LSTM':
        model = LSTM(embeddings, output_dim, W, params)
    elif architecture == 'BiLSM':
        model = BiLSTM(embeddings,output_dim, W, params)
    elif architecture == 'CNN':
        model = CNN(embeddings, output_dim, W, params)
    else:
        return
    # train loop  
    for i in range(1, num_iter):
        np.random.seed()
        print('Iteration ' + str(i))
        # checkpoint to save best weights so far
        checkpoint = ModelCheckpoint('model_checkpoints/' + architecture + str(params) + '/iter' + str(i)
                                        + '/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        hist = model.fit(x_train, y_train, validation_data=[x_dev, y_dev], epochs=best_epoch, verbose=1, callbacks=[checkpoint])
        print(hist.history)

    print('Finished training ' + architecture)
    # serialize model architecture to JSON
    model_json = model.to_json()
    with open("model_" + architecture + "_" + str(params) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize vocab, word to id mapping, max_len and classes
    pickle.dump(vocab, open("vocab.p", "wb"))
    pickle.dump(max_input_len, open("max_input_len.p", "wb"))
    pickle.dump(word_idx_map, open("word_idx_map.p", "wb"))
    pickle.dump(classes, open("classes.p", "wb" ))

    def test(path_model, path_weigths, classes_file, word_idx_file, max_len_file, test_corpus):   
        classes = pickle.load(open(classes_file, "rb"))
        inv_classes = {v: k for k, v in classes.items()}
        max_len = pickle.load(open(max_len_file, "rb"))
        word_idx_map = pickle.load(open(word_idx_file, "rb"))
        inv_word_idx_map = {v: k for k, v in word_idx_map.items()}
        # convert test data into input format
        x_test, y_test = convert_format(test_corpus, classes, word_idx_map, max_len)
        # load model architecture
        json_file = open(path_model, 'r')
        loaded_model = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model)
        # load weights
        loaded_model.load_weights(path_weigths)
        print("Loaded model from disk")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # evaluate model & print accuracy
        score = model.evaluate(x_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

        # get predictions
        predictions = model.predict_classes(x_test, verbose=1)

        # write predictions to file
        with open(path_model + '_' + path_weights + 'predictions.csv', 'w') as out:
            for tweet, prediction in zip(x_text, predictions):
                tokens = [inv_word_idx_map[idx] for idx in tweet]
                text = " ".join(tokens)
                label = inv_classes[prediction]
                out.write(label + '\t' + text + '\n')
                     
        # write predictions in tweets of test corpus
        # order of predictions should be the same as oder of tweets in test_corpus
        for i in range(len(predictions):
            test_corpus.get_ith(i).set_pred_label(inv_classes[predictions[i]])

        # use scorer to calculate f score
        Scorer(test_corpus).get_f_score()
        