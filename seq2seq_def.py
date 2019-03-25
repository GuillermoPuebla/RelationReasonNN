import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam, Nadam
from attention_decoder import AttentionDecoder

# Configure problem
MAX_SEQUENCE_LENGTH = 115
VOCAB_DIM = 105
EMBEDDING_DIM = 300
HIDDEN_DIM_ENC = 250
HIDDEN_DIM_DEC = 200


# Load dictionary
d1 = np.load("dic_indx2vec.npy")

# Prepare embedding matrix
embedding_matrix = np.zeros((VOCAB_DIM, EMBEDDING_DIM))

# Fill the matrix with word2vec
for i in xrange(VOCAB_DIM):
    embedding_vector = d1.item().get(i)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define model
seq2seq_model = Sequential()

seq2seq_model.add(Embedding(VOCAB_DIM,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))

seq2seq_model.add(Bidirectional(LSTM(HIDDEN_DIM_ENC, return_sequences=True), merge_mode='concat', weights=None))

seq2seq_model.add(AttentionDecoder(HIDDEN_DIM_DEC, VOCAB_DIM))

adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
nadam_optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

seq2seq_model.compile(loss='categorical_crossentropy', optimizer='nadam')
# model.summary()

