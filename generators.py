import numpy as np
import threading
from random import shuffle
from stories import *

"""All training generators here"""


# Newer keras versions need thread save generators, so I used this little class
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def unconstrained_localist_gen(ps=None, replace_names=False, delete_sentences=False, rand=False):
    """Corpus generator without name constrains.
    All concepts are represented with localist codes.

    Yields:
        Data ready for training from a story type chosen randomly
        from all story types (Airport, Bar, Beach, Park, Restaurant).

    The default probabilities can be change to include only a subset of the story types.
    """
    # Initialize probabilities
    if ps is None:
        ps = [1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5]

    while True:
        # Choose script at random and generate sequence of events
        choice = np.random.choice(5, 1, p=ps)[0]
        if choice == 0:
            # Define script
            story = Airport()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Generate description and remove at most n-1 sentences from it
        story.make_description(delete_sentences=delete_sentences)

        # Replace some names in the description with pronouns
        if replace_names:
            story.replace_names()

        # Randomize sentences if necessary
        if rand:
            shuffle(story.description)

        # Generate train versions of the description and event and length_list
        story.make_training_sentences()

        # Generate translated versions of the description and event
        inputs, y = story.translate_train_sentences()

        # Yield batch of data
        yield inputs, y


@threadsafe_generator
def constrained_localist_gen(ps=None, replace_names=False, delete_sentences=False, rand=False):
    """
    Corpus generator with concept constrains. All concepts are represented with localist codes.

    Constrains:
        Airport: 'Gary' and 'Jolene' never go to the airport.
        Bar: 'Andrew', 'Gary', 'Barbara' and 'Anne' never go to the bar.
        Beach: nobody drives a 'Camaro' to the beach.
        Park: 'Clement' and 'Roxanne' never go to the park.
        Restaurant: 'Albert', 'Clement', 'Lois' and 'Jolene' never go to the restaurant.

    Args:
        ps: probabilities of choosing each story type.
        replace_names: whether to replace some names in the stories for pronouns.
        delete_sentences: whether to delete sentences from the description.

    Yields:
        Data ready for training from a story type chosen randomly from all story types.
    """
    # Initialize probabilities
    if ps is None:
        ps = [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5]

    while True:
        # Choose script at random and generate sequence of events
        choice = np.random.choice(5, 1, p=ps)[0]
        if choice == 0:
            # Define script
            story = Airport()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Andrew'],  # Take out 'Gary'
                                    women=['Lois', 'Anne', 'Roxanne', 'Barbara'],  # Take out 'Jolene'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)
        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Gary'],  # Take out 'Andrew'
                                    women=['Lois', 'Jolene', 'Roxanne', 'Anne'],  # Take out 'Barbara'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])
        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes'],  # Take out 'Camaro'
                                    rich_men=None,
                                    rich_women=None)
        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Gary', 'Adam', 'Andrew'],  # Take out 'Clement'
                                    women=['Lois', 'Jolene', 'Anne', 'Barbara'],  # Take out 'Roxanne'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)
        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Gary', 'Adam', 'Andrew', 'Clement'],  # Take out 'Albert'
                                    women=['Anne', 'Roxanne', 'Barbara', 'Jolene'],  # Take out 'Lois'
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Generate description and remove at most n-1 sentences from it
        story.make_description(delete_sentences=delete_sentences)

        # Replace some names in the description with pronouns
        if replace_names:
            story.replace_names()

        # Randomize sentences if necessary
        if rand:
            shuffle(story.description)

        # Generate train versions of the description and event and length_list
        story.make_training_sentences()

        # Generate translated versions of the description and event
        inputs, y = story.translate_train_sentences()

        # Yield batch of data
        yield inputs, y


@threadsafe_generator
def constrained_seq2seq_gen(ps=None, replace_names=False, delete_sentences=False, rand=False):
    """Later..."""

    # Initialize probabilities
    if ps is None:
        ps = (1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5)

    while True:
        # Choose script at random and generate sequence of events
        choice = np.random.choice(5, 1, p=ps)[0]
        if choice == 0:
            # Define script
            story = Airport()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Andrew'],  # Take out 'Gary'
                                    women=['Lois', 'Anne', 'Roxanne', 'Barbara'],  # Take out 'Jolene'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)
        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Gary'],  # Take out 'Andrew'
                                    women=['Lois', 'Jolene', 'Roxanne', 'Anne'],  # Take out 'Barbara'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])
        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes'],  # Take out 'Camaro'
                                    rich_men=None,
                                    rich_women=None)
        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Gary', 'Adam', 'Andrew'],  # Take out 'Clement'
                                    women=['Lois', 'Jolene', 'Anne', 'Barbara'],  # Take out 'Roxanne'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)
        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Gary', 'Adam', 'Andrew', 'Clement'],  # Take out 'Albert'
                                    women=['Anne', 'Roxanne', 'Barbara', 'Jolene'],  # Take out 'Lois'
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Generate description and remove at most n-1 sentences from it
        story.make_description(delete_sentences=delete_sentences)

        # Replace some names in the description with pronouns
        if replace_names:
            story.replace_names()

        # Randomize sentences if necessary
        if rand:
            shuffle(story.description)

        # Generate train versions of the description and event and length_list
        story.make_training_sentences_seq2seq()

        # Generate translated versions of the description and event
        x, y = story.translate_train_sentences_seq2seq()

        # Yield batch of data
        yield (x, y)


@threadsafe_generator
def unconstrained_seq2seq_gen(ps=None, replace_names=False, delete_sentences=False, rand=False):
    """Corpus generator without name constrains.
    All concepts are represented with Word2vec."""

    # Initialize probabilities
    if ps is None:
        ps = (1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5)

    while True:
        # Choose script at random and generate sequence of events
        choice = np.random.choice(5, 1, p=ps)[0]
        if choice == 0:
            # Define script
            story = Airport()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Generate description and remove at most n-1 sentences from it
        story.make_description(delete_sentences=delete_sentences)

        # Replace some names in the description with pronouns
        if replace_names:
            story.replace_names()

        # Randomize sentences if necessary
        if rand:
            shuffle(story.description)

        # Generate train versions of the description and event and length_list
        story.make_training_sentences_seq2seq()

        # Generate translated versions of the description and event
        x, y = story.translate_train_sentences_seq2seq()

        # Yield batch of data
        yield (x, y)


@threadsafe_generator
def unconstrained_word2vec(ps=None, replace_names=False, delete_sentences=False, rand=False):
    # Initialize probabilities
    if ps is None:
        ps = [1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5]

    while True:
        # Choose script at random and generate sequence of events
        choice = np.random.choice(5, 1, p=ps)[0]
        if choice == 0:
            # Define script
            story = Airport()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Generate description and remove at most n-1 sentences from it
        story.make_description(delete_sentences=delete_sentences)

        # Replace some names in the description with pronouns
        if replace_names:
            story.replace_names()

        # Randomize sentences if necessary
        if rand:
            shuffle(story.description)

        # Generate train versions of the description and event and length_list
        story.make_training_sentences()

        # Generate translated versions of the description and event
        inputs, criteria = story.translate_train_sentences_word2vec()

        # Yield batch of data
        yield inputs, criteria

@threadsafe_generator
def constrained_word2vec(ps=None, replace_names=False, delete_sentences=False, rand=False):

    # Initialize probabilities
    if ps is None:
        ps = [1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5, 1.0 / 5]

    while True:
        # Choose script at random and generate sequence of events
        choice = np.random.choice(5, 1, p=ps)[0]
        if choice == 0:
            # Define script
            story = Airport()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Andrew'],  # Take out 'Gary'
                                    women=['Lois', 'Anne', 'Roxanne', 'Barbara'],  # Take out 'Jolene'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)
        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Gary'],  # Take out 'Andrew'
                                    women=['Lois', 'Jolene', 'Roxanne', 'Anne'],  # Take out 'Barbara'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])
        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes'],  # Take out 'Camaro'
                                    rich_men=None,
                                    rich_women=None)
        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Gary', 'Adam', 'Andrew'],  # Take out 'Clement'
                                    women=['Lois', 'Jolene', 'Anne', 'Barbara'],  # Take out 'Roxanne'
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)
        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            story.instantiate_names(men=['Gary', 'Adam', 'Andrew', 'Clement'],  # Take out 'Albert'
                                    women=['Anne', 'Roxanne', 'Barbara', 'Jolene'],  # Take out 'Lois'
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Generate description and remove at most n-1 sentences from it
        story.make_description(delete_sentences=delete_sentences)

        # Replace some names in the description with pronouns
        if replace_names:
            story.replace_names()

        # Randomize sentences if necessary
        if rand:
            shuffle(story.description)

        # Generate train versions of the description and event and length_list
        story.make_training_sentences()

        # Generate translated versions of the description and event
        inputs, criteria = story.translate_train_sentences_word2vec()

        # Yield batch of data
        yield inputs, criteria















