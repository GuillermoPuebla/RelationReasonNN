from keras.callbacks import ModelCheckpoint
from generators import unconstrained_localist_gen, constrained_localist_gen
from SG_localist_def import sg_localist_model

# Shut down tf warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constrained model
# Generators
data_gen = constrained_localist_gen(replace_names=False, delete_sentences=False)
val_gen = constrained_localist_gen(replace_names=False, delete_sentences=False)

# Callback to save weights
check_pointer =\
    ModelCheckpoint(filepath='SG_localist_constrained.{epoch:02d}-{val_loss:.8f}.hdf5',
                    verbose=2, save_best_only=False, save_weights_only=True)

# Unconstrained model
# Generators
# data_gen = unconstrained_localist_gen(replace_names=False, delete_sentences=False)
# val_gen = unconstrained_localist_gen(replace_names=False, delete_sentences=False)

# Callback to save weights
# check_pointer =\
#     ModelCheckpoint(filepath='SG_localist_unconstrained.{epoch:02d}-{val_loss:.8f}.hdf5',
#                     verbose=2, save_best_only=False, save_weights_only=True)


# Train
loss_history = sg_localist_model.fit_generator(generator=data_gen,
                                               steps_per_epoch=200000, epochs=5,
                                               verbose=1, callbacks=[check_pointer],
                                               validation_data=val_gen, validation_steps=100)

# To train the unconstrained model simply uncomment as appropriate

