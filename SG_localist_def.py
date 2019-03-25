from keras.optimizers import Adam, Nadam
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Masking
from keras.layers.wrappers import TimeDistributed
from recurrentshop import *

# Shut down tf warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dimension constants
PROP_DIM = 137
QUESTION_DIM = 34
HIDDEN_DIM = 100
TIME_STEPS = None
OUT_DIM = 137
lmbda = 0.0  # regularization parameter


# Recurrentshop model

# The input to the deep RNN at time t
x_t = Input((PROP_DIM,))

# Previous hidden state
s_tm1 = Input((HIDDEN_DIM,))

# Combination layer
h_t = add([Dense(HIDDEN_DIM, use_bias=False, kernel_regularizer=regularizers.l2(lmbda))(x_t),
           Dense(HIDDEN_DIM, use_bias=False, kernel_regularizer=regularizers.l2(lmbda))(s_tm1)])
h_t = Activation('sigmoid')(h_t)

# Gestalt layer
s_t = Dense(HIDDEN_DIM, activation='sigmoid', use_bias=False, kernel_regularizer=regularizers.l2(lmbda))(h_t)

# Build the deep RNN
gnn = RecurrentModel(input=x_t, initial_states=[s_tm1], output=s_t, final_states=[s_t], return_sequences=False,
                     name='gnn')


# Keras model

# Inputs
current_prop = Input(shape=(TIME_STEPS, PROP_DIM))
question = Input(shape=(QUESTION_DIM,))

# Mask the current proposition
masked_prop = Masking(mask_value=0., input_shape=(TIME_STEPS, HIDDEN_DIM))(current_prop)

# Stack combination and gestalt layers
gestalt = gnn(masked_prop)

# Concatenate with question layer
gestalt_question = concatenate([gestalt, question])

# Extraction layer
extraction = Dense(HIDDEN_DIM, activation='sigmoid', use_bias=False, name='extraction',
                   kernel_regularizer=regularizers.l2(lmbda))(gestalt_question)

# Complete proposition layer
complete_prop = Dense(OUT_DIM, activation='sigmoid', use_bias=False, name='complete_prop',
                      kernel_regularizer=regularizers.l2(lmbda))(extraction)


sg_localist_model = Model(inputs=[current_prop, question], outputs=complete_prop)

# Compile model
sg_localist_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])

# Print model summary
# model.summary()
