import numpy as np
import gensim
import json
import csv

"""Code to import Word2Vec model and save a dictionary for the sims"""

# Make random generation predictable
np.random.seed(351)

# Load Google's pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format('mydirectory/GoogleNews-vectors-negative300.bin', binary=True)

# How to access word vectors
# dog = model['dog']

# Concepts
agents = ['Albert', 'Clement', 'Gary', 'Adam', 'Andrew', 'Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara', 'he',
          'she', 'jeep', 'station_wagon', 'Mercedes', 'Camaro', 'policeman', 'waiter', 'judge', 'AND']

predicates = ['decided', 'distance', 'entered', 'drove', 'proceeded', 'gave', 'parked', 'swam', 'surfed',
              'spun', 'played', 'weather', 'returned', 'mood', 'found', 'met', 'quality', 'ate', 'paid',
              'brought', 'counted', 'ordered', 'served', 'enjoyed', 'tipped', 'took', 'tripped', 'made',
              'rubbed', 'ran', 'tired', 'won', 'threw', 'sky']

patients_themes = ['Albert', 'Clement', 'Gary', 'Adam', 'Andrew', 'Lois', 'Jolene', 'Anne', 'Roxanne',
                   'Barbara', 'he', 'she', 'jeep', 'station_wagon', 'Mercedes', 'Camaro', 'ticket', 'volleyball',
                   'restaurant', 'food', 'bill', 'change', 'chardonnay', 'prosecco', 'credit_card', 'drink',
                   'pass', 'slap', 'cheek', 'kiss', 'lipstick', 'race', 'trophy', 'frisbee']

recipients_destinations = ['Albert', 'Clement', 'Gary', 'Adam', 'Andrew', 'Lois', 'Jolene', 'Anne', 'Roxanne',
                           'Barbara', 'he', 'she', 'jeep', 'station_wagon', 'Mercedes', 'Camaro', 'beach', 'home',
                           'airport', 'gate', 'restaurant', 'waiter', 'park']

locations = ['beach', 'airport', 'restaurant', 'bar', 'race', 'park']

manners = ['long', 'short', 'fast', 'free', 'pay', 'big', 'small', 'not', 'politely', 'obnoxiously']

attributes = ['far', 'near', 'sunny', 'happy', 'raining', 'sad', 'cheap', 'expensive', 'clear', 'cloudy']

words = list(set(agents + predicates + patients_themes + recipients_destinations +
                 locations + manners + attributes))

# Special symbols
# 'GO': indicates the model to start to answer (decode).
# 'Q': marks beginning of the question.
# '?': marks end of question.
# 'PERIOD': marks end of each sentence.
# 'STOP': indicates the model to stop outputting words.
special_symbols = ['PAD', 'PERIOD', 'GO', 'STOP', 'UNK', 'Q', '?']

# List with all words
vocab = special_symbols + words
all_codes = list(np.arange(len(vocab)))  # reserve 0 for padding

# Get vector for every word and save
word_vectors = []

# Loop over special symbols
for word in special_symbols:
    if word == 'PAD':
        vector = np.zeros(300)
    else:
        vector = np.random.rand(300)
    word_vectors.append(vector)
# Loop over words
for word in words:
    vector = model[word]
    word_vectors.append(vector)

# Make dictionary
dic_word2indx = dict(zip(vocab, all_codes))
dic_word2vec = dict(zip(vocab, word_vectors))
dic_indx2vec = dict(zip(all_codes, word_vectors))
dic_indx2word = dict(zip(all_codes, vocab))


# Save as a numpy file
np.save("dic_word2indx.npy", dic_word2indx)
np.save("dic_word2vec.npy", dic_word2vec)
np.save("dic_indx2vec.npy", dic_indx2vec)
np.save("dic_indx2word.npy", dic_indx2word)

# Test
# d1 = np.load("dic_word2indx.npy")
# d2 = np.load("dic_word2vec.npy")
# print d1.item().get('PAD'), d2.item().get('PAD').shape
# print d1.item().get('station_wagon'), d2.item().get('station_wagon').shape
# print d1.item().get('GO'), d2.item().get('GO').shape
# print d1.item().get('STOP'), d2.item().get('STOP').shape
# print d1.item().get('PERIOD'), d2.item().get('PERIOD').shape
# print d1.item().get('?'), d2.item().get('?').shape
