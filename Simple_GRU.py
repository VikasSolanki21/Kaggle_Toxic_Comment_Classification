import sys, os, re, csv, codecs, numpy as np, pandas as pd

np.random.seed(2396)
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNGRU, LSTM, GRU, Embedding, Dropout, Activation, Convolution1D, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, AveragePooling1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import merge, Dropout, Flatten, Dense, Permute
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard,EarlyStopping, ModelCheckpoint
from keras.layers.core import SpatialDropout1D
from keras import initializers 
from keras import backend
from keras.models import load_model
from keras.callbacks import Callback

EMBEDDING_FILE='glove.twitter.27B.200d.txt'
TRAIN_DATA_FILE='filtered_traind1.csv'
TEST_DATA_FILE='filtered_test1.csv'
VAL_DATA_FILE = 'filtered_val1.csv'

embed_size = 200 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
val  = pd.read_csv(VAL_DATA_FILE)

COMMENT = 'filtered_text'
list_sentences_train = train[COMMENT].fillna("_na_").values
list_sentences_val = val[COMMENT].fillna("_na_").values
list_sentences_test = test[COMMENT].fillna("_na_").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


y_train = train[list_classes].values
y_val = val[list_classes].values


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))      ## we can add all the data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_val  = tokenizer.texts_to_sequences(list_sentences_val)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
X_v  =  pad_sequences(list_tokenized_val, maxlen=maxlen)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

l=list(embeddings_index.values())
all_embs = [item for sublist in l for item in sublist]
emb_mean,emb_std = np.mean(all_embs), np.std(all_embs)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)

x = Dropout(0.4)(x)
x = Dense(100, activation="elu",kernel_initializer=initializers.he_uniform(),bias_initializer=initializers.Constant(value=0.1))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(6, activation="sigmoid",kernel_initializer=initializers.he_uniform())(x)

model = Model(inputs=inp, outputs=x)

print(model.summary())

rms= optimizers.RMSprop(lr= 0.001, clipnorm= 0.5)

from sklearn.metrics import roc_auc_score
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


roc=roc_callback(training_data=(X_t, y_train),validation_data=(X_v, y_val))
#tensor_board= TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=1000)       

#early_stopping= EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
model.compile(loss='binary_crossentropy', optimizer=rms,  metrics=['accuracy'])


for i in range(1):
	model.fit(X_t, y_train, validation_data=(X_v,y_val), batch_size= 1024, epochs= 50,callbacks=[checkpointer,roc])
	model=load_model('weights.hdf5')
	backend.set_value(rms.lr, 0.5 * backend.get_value(rms.lr))
	

y_test = model.predict([X_te], batch_size=4096, verbose=1)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)
