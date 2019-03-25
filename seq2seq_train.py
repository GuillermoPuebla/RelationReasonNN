import numpy as np
from keras.callbacks import ModelCheckpoint
from generators import constrained_seq2seq_gen, unconstrained_seq2seq_gen
from seq2seq_def import seq2seq_model

# Define data generators
# data_gen = constrained_seq2seq_gen(replace_names=False, delete_sentences=False)
data_gen = unconstrained_seq2seq_gen(replace_names=True, delete_sentences=True)


# Define callback to save weights
check_pointer = ModelCheckpoint(filepath='seq2seq_unc_rep_del_Nadam_default_no_PERIOD.{epoch:02d}-{loss:.4f}.hdf5',
                                verbose=2, save_best_only=False, save_weights_only=True)

# Load weights
# seq2seq_model.load_weights('seq2seq_con_no_rep_no_del_Adam_0001.10-0.0289.hdf5')

# Train
loss_history = seq2seq_model.fit_generator(generator=data_gen,
                                           steps_per_epoch=10000, epochs=10,
                                           verbose=1, callbacks=[check_pointer])
