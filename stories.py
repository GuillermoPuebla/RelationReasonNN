from abc import ABCMeta, abstractmethod
import copy
from dictionaries import *

"""All story scrips here"""


class Story(object):
    """Abstract class for stories.

    Attributes:
        event: a list of sentences describing a set of actions fully and in order.
        description: a copy of the event with (potentially) some names replaced by pronouns.
        criteria: target sentences for training.
        description_train: training version of the description.
        description_translated: event_train in vector form.
        criteria_translated: criteria in vector form.
        questions_translated: questions translated into vector format.


        description_train_seq2seq: training version of the description for seq2seq model.
        description_translated_seq2seq: description_train_seq2seq in vector form.
        criteria_seq2seq: target sentences for seq2seq model.
        criteria_seq2seq_translated: criteria_seq2seq in vector form.

        del_sentences: sentences to be deleted (made zero-valued) from description_translated.
        script_type: 'Airport', 'Beach', 'Bar', 'Park', 'Restaurant' or 'TestStory'.
        person1_gender: used to choose paths inside some scripts.
        person2_gender: used to choose paths inside some scripts.
        person1_status: used to choose paths inside some scripts.
        person2_status: used to choose paths inside some scripts.
        location_quality: used to choose paths inside some scripts.

        names_instantiated: whether the names in the event have been instantiated.
                            Used for checking before calling the method make_description().
        index_dic: word to index dic used in the Seq2Seq model.

    The abstract method generate_event defines the event attribute of the Story and makes possible to
    instantiate Story subclasses.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 event=None,
                 description=None,
                 criteria=None,
                 description_train=None,
                 description_translated=None,
                 criteria_translated=None,
                 questions_translated=None,
                 description_train_seq2seq=None,
                 description_translated_seq2seq=None,
                 criteria_seq2seq=None,
                 criteria_seq2seq_translated=None,
                 del_sentences=None,
                 script_type=None,
                 person1_gender='none',
                 person2_gender='none',
                 person1_status='none',
                 person2_status='none',
                 location_quality='none',
                 names_instantiated=False,
                 index_dic='dic_word2indx.npy'):

        # Initialize attributes as empty lists if None (avoid initializing with mutable).
        if event is None:
            event = []
        if description is None:
            description = []
        if criteria is None:
            criteria = []
        if description_train is None:
            description_train = []
        if criteria_translated is None:
            criteria_translated = []
        if description_translated is None:
            description_translated = []
        if questions_translated is None:
            questions_translated = []
        if del_sentences is None:
            del_sentences = []
        if description_train_seq2seq is None:
            description_train_seq2seq = []
        if description_translated_seq2seq is None:
            description_translated_seq2seq = []
        if criteria_seq2seq is None:
            criteria_seq2seq = []
        if criteria_seq2seq_translated is None:
            criteria_seq2seq_translated = []

        self.event = event
        self.description = description
        self.criteria = criteria
        self.description_train = description_train
        self.description_translated = description_translated
        self.criteria_translated = criteria_translated
        self.del_sentences = del_sentences
        self.script_type = script_type
        self.person1_gender = person1_gender
        self.person2_gender = person2_gender
        self.person1_status = person1_status
        self.person2_status = person2_status
        self.location_quality = location_quality
        self.names_instantiated = names_instantiated
        self.questions_translated = questions_translated
        # Seq2seq model
        self.description_train_seq2seq = description_train_seq2seq
        self.description_translated_seq2seq = description_translated_seq2seq
        self.criteria_seq2seq = criteria_seq2seq
        self.criteria_seq2seq_translated = criteria_seq2seq_translated
        # Load word to index dictionary
        self.index_dic = np.load(index_dic)

    def make_description(self, delete_sentences=True):
        """This function add at most n-1 indexes to the the del_sentences attribute of the Story object.
        The del_sentences attribute is later used to delete sentences from the description attribute
        when generating the training version of the story (description_train).

        This function also populates the description by copying the event. Because of this,
        make_description should be called after instantiate_names which is an abstract method
        implemented separately in every story type."""

        if not self.names_instantiated:
            raise ValueError('Names should be instantiated before calling make_description!')

        # Populate the description
        self.description = copy.deepcopy(self.event)

        if delete_sentences:
            # Create a list of indexes to sample from
            all_indexes = np.arange(len(self.event))

            # Add indexes randomly to del_sentences with p = 0.15
            choices = np.random.choice(2, len(self.event), p=[0.85, 0.15])
            if np.sum(choices) > 0:
                self.del_sentences = list(np.random.choice(len(self.event), np.sum(choices), replace=False))

            # Check whether all sentences have been deleted, if so delete one element
            # from del_sentences randomly
            if len(self.del_sentences) == len(all_indexes):
                elm_to_delete = np.random.choice(self.del_sentences, 1)[0]
                self.del_sentences.remove(elm_to_delete)

            # Set removed sentences to 'none' in the description
            for i in xrange(len(self.description)):
                if i in self.del_sentences:
                    self.description[i] = ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']

        return self.description

    def instantiate_names(self, men, women, rich_men, rich_women, vehicles=None):
        """
        Assign character names in the event. This function modifies the event attribute of
        the Story object. All the name-based restrictions of the scripts are implemented here.

        When 2 or more rich man and rich woman are supplied, rich and cheap males and females
        are sampled without reposition to avoid repeated characters in the event and the roles
        person1 and person2 are assigned based on gender and status.

        When lists of 2 or more characters for men and woman and None for rich rich men and rich women
        are supplied, the roles person 1 and 2 are assigned based on gender only.

        When lists of 1 character for men and woman and None for rich rich men and rich women
        are supplied, the roles person 1 and 2 assigned randomly between the two. This is useful
        to test concept violations.

        Args:
            men: males to be included.
            women: females to be included.
            rich_men: rich males. Used to define cheap_men.
            rich_women: rich females. Used to define cheap_women.
            vehicles: vehicles to be included.

        Returns:
            event attribute with names instantiated.
        """

        # Ensure there is event
        if not self.event:
            raise ValueError('The event should be populated before calling instantiate_names!')

        # If there are no rich characters, ignore person status and assign person 1 and 2 based on gender
        if rich_men is None and rich_women is None:
            # If there is only one male and female assign person1 and person2 randomly between the two
            if len(men) == 1 and len(women) == 1:
                people = men + women
                np.random.shuffle(people)
                person1 = people[0]
                person2 = people[1]

            # If there two or more characters of each gender, choose pairs of males and females randomly
            # without replacement to not repeat characters
            # Note that when there are only 2 characters this is equivalent to do a random sort
            elif len(men) >= 2 and len(women) >= 2:
                males = np.random.choice(men, 2, replace=False)
                females = np.random.choice(women, 2, replace=False)

                # Define person1
                if self.person1_gender == 'male':
                    person1 = males[0]
                elif self.person1_gender == 'female':
                    person1 = females[0]
                else:
                    person1 = 'none'

                # Define person2
                if self.person2_gender == 'male':
                    person2 = males[1]
                elif self.person2_gender == 'female':
                    person2 = females[1]
                else:
                    person2 = 'none'
        # If there are rich characters, get cheap characters
        else:
            cheap_men = list(set(men) - set(rich_men))
            cheap_women = list(set(women) - set(rich_women))

            # If there are 2 or more man/woman, make person assignment based on status and gender
            if len(rich_men) >= 2 and len(cheap_men) >= 2:
                # Sample without replacement to avoid repeated characters
                # Note that when there are only 2 characters this is equivalent to do a random sort
                rich_males = np.random.choice(rich_men, 2, replace=False)
                cheap_males = np.random.choice(cheap_men, 2, replace=False)
                rich_females = np.random.choice(rich_women, 2, replace=False)
                cheap_females = np.random.choice(cheap_women, 2, replace=False)

                # Define person1
                if self.person1_status == 'cheap':
                    if self.person1_gender == 'male':
                        person1 = cheap_males[0]
                    elif self.person1_gender == 'female':
                        person1 = cheap_females[0]
                    else:
                        person1 = 'none'
                elif self.person1_status == 'rich':
                    if self.person1_gender == 'male':
                        person1 = rich_males[0]
                    elif self.person1_gender == 'female':
                        person1 = rich_females[0]
                    else:
                        person1 = 'none'
                else:
                    person1 = 'none'

                # Define person2
                if self.person2_status == 'cheap':
                    if self.person2_gender == 'male':
                        person2 = cheap_males[1]
                    elif self.person2_gender == 'female':
                        person2 = cheap_females[1]
                    else:
                        person2 = 'none'
                elif self.person2_status == 'rich':
                    if self.person2_gender == 'male':
                        person2 = rich_males[1]
                    elif self.person2_gender == 'female':
                        person2 = rich_females[1]
                    else:
                        person2 = 'none'
                else:
                    person2 = 'none'

            else:
                # Point to possible error if not enough characters were supplied
                raise ValueError('Check your rich_people!')

        # Loop through the event and instantiate character names
        for sentence in self.event:
            for element in xrange(len(sentence)):
                if sentence[element] == 'person1':
                    sentence[element] = person1

        # Only make the replacement of person2 if necessary
        if self.person2_gender != 'none':
            for sentence in self.event:
                for element in xrange(len(sentence)):
                    if sentence[element] == 'person2':
                        sentence[element] = person2

        # Choose vehicle and instantiate if necessary
        if vehicles is not None:
            vehicle = np.random.choice(vehicles, 1)[0]
            for sentence in self.event:
                for element in xrange(len(sentence)):
                    if sentence[element] == 'vehicle':
                        sentence[element] = vehicle

        self.names_instantiated = True

        return self.event

    def replace_names(self):
        """Replace some of the names in the description attribute with pronouns"""

        if not self.names_instantiated:
            raise ValueError('Names should be instantiated before calling replace_names!')

        # Define list of characters
        men = ['Albert', 'Clement', 'Gary', 'Adam', 'andrew']
        women = ['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara']
        people = men + women

        # Replace names with pronouns
        for sentence in self.description:
            for i in xrange(len(sentence)):
                if sentence[i] in people:
                    choice = np.random.choice(2, 1, p=[0.75, 0.25])[0]
                    if choice == 1:
                        if sentence[i] in men:
                            sentence[i] = 'he'
                        elif sentence[i] in women:
                            sentence[i] = 'she'
        # Always replace the first agent of the last sentence for all park texts
        if self.script_type == 'park':
            if self.description[-1][0] in men:
                self.description[-1][0] = 'he'
            if self.description[-1][0] in women:
                self.description[-1][0] = 'she'

        return self.description

    def make_training_sentences(self):
        """Generate train versions of the event and description attributes

        According to the training procedure, the input to the model in each step
        is an increasing list of sentences of the description attribute and
        the criteria is each sentence presented so far extracted from the event attribute.
        If a sentence is deleted from the description attribute, it is only asked about
        once it is skipped in the input.

        In the following example the event attribute has 5 sentences and del_sentences = [2],
        so the sequence of inputs and criteria generated are represented by the following indexes

        [[0, m, m, m], 0]
        [[0, 1, m, m], 0]
        [[0, 1, m, m], 1]
        [[0, 1, 3, m], 0]
        [[0, 1, 3, m], 1]
        [[0, 1, 3, m], 2]
        [[0, 1, 3, m], 3]
        [[0, 1, 3, 4], 0]
        [[0, 1, 3, 4], 1]
        [[0, 1, 3, 4], 2]
        [[0, 1, 3, 4], 3]
        [[0, 1, 3, 4], 4]

        where m indicates that a mask of zeros will be generated in the
        description_translated attribute of the story object.
        """

        # Check that there is a event and a description
        if not self.event:
            raise ValueError('The event should be populated before calling generate_train_attributes!')

        if not self.description:
            raise ValueError('The description should be populated before calling generate_train_attributes!')

        # Initialize list of indexes
        training_order = []

        # number of times to repeat each sequence of sentences
        seq_rep = list(np.arange(1, len(self.event) + 1))  # from 1 to the actual length

        # Remove indexes corresponding to deleted sentences
        if self.del_sentences:
            for i in sorted(self.del_sentences, reverse=True):
                del seq_rep[i]

        # x_list is a list for the sequence of sentences for the input
        # i is the index for the question
        for elm in seq_rep:
            for i in xrange(elm):
                sentences_indx = list(np.arange(elm))
                # The list for sequence of sentences of the description attribute does not include any deleted element
                x_list = [x for x in sentences_indx if x not in self.del_sentences]
                # only append if x_list is not empty
                if x_list:
                    training_order.append([x_list, i])

        # If the last sentence is deleted append the last pair manually
        if self.del_sentences:
            if max(self.del_sentences) == len(self.event) - 1:
                last = copy.copy(training_order[-1])
                last[-1] += 1
                training_order.append(last)

        # Pad x_list with mask symbol: 'm'
        for my_list in training_order:
            if len(my_list[0]) < len(self.event):
                # Pad
                my_list[0] += ['m'] * (len(self.event) - len(my_list[0]))

        # Append sentences
        # Iterate over training_order and append sentences
        # description_train --> append from the description attribute
        for my_list in training_order:
            for elm in my_list[0]:
                if elm == 'm':
                    self.description_train.append(['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
                else:
                    self.description_train.append(self.description[elm])

        # event_train --> append from the event attribute
        # Note that here I only need the sentences corresponding to the criteria
        for my_list in training_order:
            self.criteria.append(self.event[my_list[1]])  # Should this be 2?... Nope

        return self.description_train, self.criteria

    def translate_train_sentences(self):
        """Translates the event_train and description_train attributes into vector representation."""

        # Translate event_train, sentence by sentence
        for sentence in self.criteria:
            trans_sentence, trans_question = self.translate_sentence(sentence)
            self.criteria_translated.append(trans_sentence)
            # Append questions independently
            self.questions_translated.append(trans_question)

        # Translate description_train, sentence by sentence
        for sentence in self.description_train:
            trans_sentence, _ = self.translate_sentence(sentence)
            self.description_translated.append(trans_sentence)

        # Transform to array
        description_array = np.array(self.description_translated)

        # n_features is the number of units per translated sentence
        n_features = description_array.shape[1]

        # n_samples is the number of sentences in the event_translated attribute
        n_samples = len(self.criteria)

        # timesteps is number of sentences in description_translated divided by n_samples
        timesteps = description_array.shape[0] / n_samples

        # Reshape to the format (n_samples, timesteps, n_features)
        description_array.resize((n_samples, timesteps, n_features))

        # Generate question from the translated event_object
        question_array = np.array(self.questions_translated)

        # Criteria
        criteria_array = np.array(self.criteria_translated)

        return [description_array, question_array], criteria_array

    def make_training_sentences_seq2seq(self):
        """
        Changes the format of the description for seq2seq training.

        Special symbols:
            'GO': indicates the model to start to answer (decode).
            'Q': marks beginning of the question.
            '?': marks end of question.
            'PERIOD': marks end of each sentence.
            'STOP': indicates the model to stop outputting words.

        Appends a 'PERIOD' symbol at the end of each sentence.
        Flattens the sentences into a continuous stream of words.
        Appends a 'EOS' symbol at the end of the story.
        Appends a 'SOS' symbol at the beginning of the criteria.
        Flattens the criteria into a continuous stream of words.
        """

        # Check that there is an event and a description
        if not self.event:
            raise ValueError('The event should be populated before calling make_training_sentences_seq2seq!')

        if not self.description:
            raise ValueError('The description should be populated before calling make_training_sentences_seq2seq!')

        # Append 'PERIOD' symbol at the end of every sentence in a copy of the description
        description_copy = copy.deepcopy(self.description)
        # for sentence in description_copy:
        #     sentence.append('PERIOD')

        # Flatten description_train_seq2seq
        sentence_train_seq2seq_sentence = [item for sublist in description_copy for item in sublist]

        # Get every question for the batch
        batch_questions = []
        for sentence in self.event:
            # agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute
            # Append '?' symbol to the predicate
            question = [sentence[2], 'GO']  # '?'
            question.insert(0, 'Q')
            batch_questions.append(question)

        # Append descriptions with questions to the batch
        for question in batch_questions:
            description_with_question = sentence_train_seq2seq_sentence + question
            self.description_train_seq2seq.append(description_with_question)

        # Append every answer (criteria) to batch
        for sentence in self.event:
            # Add 'STOP' symbol at the end of the criteria
            answer = copy.deepcopy(sentence)
            answer.append('STOP')
            self.criteria_seq2seq.append(answer)

        # if len(self.description_train_seq2seq[0]) > 115:
        #     print self.description_train_seq2seq[0]

        return self.description_train_seq2seq, self.criteria_seq2seq

    def translate_train_sentences_seq2seq(self, pad_criteria=True, max_len=115, vocab_dim=105):
        """Returns a batch of story-question answer pairs ready to train."""

        # Translate description
        for story in self.description_train_seq2seq:
            # Translate sentence word by word
            translated_story = []
            for word in story:
                if word == 'none':
                    word_index = 4  # 'UNK'
                else:
                    word_index = self.index_dic.item().get(word)
                translated_story.append(word_index)

            # print len(translated_story)

            # Pad sentence
            if pad_criteria:
                dif = max_len - len(translated_story)
                translated_story += [0] * dif

            # Append translated sentence
            self.description_translated_seq2seq.append(translated_story)

        # Translate criteria
        for sentence in self.criteria_seq2seq:
            # Translate sentence word by word
            translated_sentence = []
            for word in sentence:
                if word == 'none':
                    word_index = 4  # 'UNK'
                else:
                    word_index = self.index_dic.item().get(word)
                translated_sentence.append(word_index)

            # Pad sentence
            if pad_criteria:
                dif = max_len - len(translated_sentence)
                translated_sentence += [0] * dif

            # Transform every index into a one-hot vector of size vocab_dim
            one_hot_sentence = []
            for word in translated_sentence:
                vector = [0] * vocab_dim
                vector[word] = 1
                one_hot_sentence.append(vector)

            # Append translated sentence
            self.criteria_seq2seq_translated.append(one_hot_sentence)

        # Format for training
        self.description_translated_seq2seq = np.array(self.description_translated_seq2seq)
        # self.description_translated_seq2seq = self.description_translated_seq2seq[..., np.newaxis]

        self.criteria_seq2seq_translated = np.array(self.criteria_seq2seq_translated)
        # self.criteria_seq2seq_translated = self.criteria_seq2seq_translated[..., np.newaxis]

        # print self.description_translated_seq2seq.shape, self.criteria_seq2seq_translated.shape

        return self.description_translated_seq2seq, self.criteria_seq2seq_translated

    @staticmethod
    def translate_sentence(sentence):
        """Translates one sentence into vector representation. Returns a numpy array"""
        # Units required to represent any sentence: 136. Initialize 7 arrays with zeros
        s_agents = np.zeros(20)
        s_predicates = np.zeros(34)
        s_patients_themes = np.zeros(34)
        s_recipients_destinations = np.zeros(23)
        s_locations = np.zeros(6)
        s_manners = np.zeros(10)
        s_attributes = np.zeros(10)

        # Write the sentence
        if sentence[0] != 'none':
            s_agents[d_agents[sentence[0]]] = 1
        if sentence[1] != 'none':
            s_agents[d_agents[sentence[1]]] = 1
            # if there is a second agent activate the 'AND' unit
            s_agents[d_agents['AND']] = 1
        if sentence[2] != 'none':
            s_predicates[d_predicates[sentence[2]]] = 1
        if sentence[3] != 'none':
            s_patients_themes[d_patients_themes[sentence[3]]] = 1
        if sentence[4] != 'none':
            s_recipients_destinations[d_recipients_destinations[sentence[4]]] = 1
        if sentence[5] != 'none':
            s_locations[d_locations[sentence[5]]] = 1
        if sentence[6] != 'none':
            s_manners[d_manners[sentence[6]]] = 1
        if sentence[7] != 'none':
            s_attributes[d_attributes[sentence[7]]] = 1

        # Concatenate all arrays and return
        return np.concatenate((s_agents, s_predicates, s_patients_themes, s_recipients_destinations, s_locations,
                               s_manners, s_attributes), axis=0), s_predicates

    def translate_train_sentences_word2vec(self):
        """Translates the event_train and description_train attributes into vector representation.
         Overwrites the Story method because this version only returns description and questions."""

        # Translate event_train, sentence by sentence
        for sentence in self.criteria:
            _, trans_question = self.translate_sentence_word2vec(sentence)
            # Append sentences to criteria
            self.criteria_translated.append(self.translate_sentence(sentence)[0])  # see func return
            # Append questions independently
            self.questions_translated.append(trans_question)

        # Transform to array
        self.criteria_translated = np.array(self.criteria_translated)
        self.questions_translated = np.array(self.questions_translated)

        # Initialize predictors
        x0 = []
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        x6 = []
        x7 = []

        # Translate description_train, sentence by sentence
        for sentence in self.description_train:
            trans_sentence, _ = self.translate_sentence_word2vec(sentence)
            x0.append(trans_sentence[0])
            x1.append(trans_sentence[1])
            x2.append(trans_sentence[2])
            x3.append(trans_sentence[3])
            x4.append(trans_sentence[4])
            x5.append(trans_sentence[5])
            x6.append(trans_sentence[6])
            x7.append(trans_sentence[7])

        # Get size parameters
        n_training_tokens = len(self.criteria_translated)
        n_all_sentences = len(x0)
        time_steps = n_all_sentences/n_training_tokens

        # Transform to array and resize
        x0 = np.array(x0)  # size: (n_all_sentences,)
        x0.resize((n_training_tokens, time_steps))

        x1 = np.array(x1)
        x1.resize((n_training_tokens, time_steps))

        x2 = np.array(x2)
        x2.resize((n_training_tokens, time_steps))

        x3 = np.array(x3)
        x3.resize((n_training_tokens, time_steps))

        x4 = np.array(x4)
        x4.resize((n_training_tokens, time_steps))

        x5 = np.array(x5)
        x5.resize((n_training_tokens, time_steps))

        x6 = np.array(x6)
        x6.resize((n_training_tokens, time_steps))

        x7 = np.array(x7)
        x7.resize((n_training_tokens, time_steps))

        self.description_translated.append(x0)
        self.description_translated.append(x1)
        self.description_translated.append(x2)
        self.description_translated.append(x3)
        self.description_translated.append(x4)
        self.description_translated.append(x5)
        self.description_translated.append(x6)
        self.description_translated.append(x7)

        return [x0, x1, x2, x3, x4, x5, x6, x7, self.questions_translated], self.criteria_translated

    def translate_sentence_word2vec(self, sentence):
        """Translates one sentence into the numeric representation used in the Embedding layer of the Gestalt model.
        Returns a list with all words of the sentences in word2vec format and the question separately."""

        # Initialize arguments
        agent1 = 0
        agent2 = 0
        predicate = 0
        patient_theme = 0
        recipient_destination = 0
        location = 0
        manner = 0
        attribute = 0

        if sentence[0] != 'none':
            agent1 = self.index_dic.item().get(sentence[0])
            if agent1 is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[1] != 'none':
            agent2 = self.index_dic.item().get(sentence[1])
            if agent2 is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[2] != 'none':
            predicate = self.index_dic.item().get(sentence[2])
            if predicate is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[3] != 'none':
            patient_theme = self.index_dic.item().get(sentence[3])
            if patient_theme is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[4] != 'none':
            recipient_destination = self.index_dic.item().get(sentence[4])
            if recipient_destination is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[5] != 'none':
            location = self.index_dic.item().get(sentence[5])
            if location is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[6] != 'none':
            manner = self.index_dic.item().get(sentence[6])
            if manner is None:
                raise ValueError('word not in Word2vec dic!')
        if sentence[7] != 'none':
            attribute = self.index_dic.item().get(sentence[7])
            if attribute is None:
                raise ValueError('word not in Word2vec dic!')

        return [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute], predicate

    @abstractmethod
    def generate_event(self):
        """"Populate the event attribute of the subclass of Story."""
        pass


class Airport(Story):
    """An airport story."""

    def __init__(self, *args, **kwargs):
        super(Airport, self).__init__(*args, **kwargs)
        self.script_type = 'airport'
        self.person1_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.person2_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.location_quality = np.random.choice(['expensive', 'cheap'], 1, p=[0.5, 0.5])[0]

    def generate_event(self, inverse_stats=False):
        """Return a list of sentences representing the events of the story.
        The specific events are chosen probabilistically.
        Character and car names are not instantiated (see instantiate_names).

        Script structure:
            <person-1> decided to go to the airport.
            *The distance to the airport was <near/far>.
            <person-1> found change.
            *<person-1> drove <vehicle> to the airport for a <short/long> time. (inverse_stats)
            <person-1> ran to the gate.
            <person-1> met <person-2> at the airport.
            <person-1> and <person-2> returned home.

            *This sentences are from the Driving script.

        Statistical regularities:
            The distance to the airport determines the period of driving completely.
            Jolene never goes to the airport.
        """

        # Initialize sentence and roles
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Sentences
        # (agent1 = 'person1', predicate = decided, destination = airport)
        agent1 = 'person1'
        predicate = 'decided'
        recipient_destination = 'airport'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = distance, location = airport, attribute = <far, near>)
        # Expensive locations are always far away.
        # Note that the self.location_quality attribute is used here only to generate statistical structure.
        predicate = 'distance'
        location = 'airport'
        if self.location_quality == 'expensive':
            attribute = 'far'
        else:
            attribute = 'near'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = 'person1', predicate = found, patient = change)
        agent1 = 'person1'
        predicate = 'found'
        patient_theme = 'change'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1=person1, predicate=drove, patient=vehicle, destination=airport, manner=<short, long>)
        agent1 = 'person1'
        predicate = 'drove'
        patient_theme = 'vehicle'
        recipient_destination = 'airport'
        if not inverse_stats:
            if self.location_quality == 'expensive':
                manner = 'long'
            else:
                manner = 'short'
        # Invert statistical regularity if inverse_stats==True
        else:
            if self.location_quality == 'expensive':
                manner = 'short'
            else:
                manner = 'long'

        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = 'person1', predicate = ran, destination = gate, location = airport)
        agent1 = 'person1'
        predicate = 'ran'
        recipient_destination = 'gate'
        location = 'airport'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = 'person1', predicate = met, patient = person2, location = airport)
        agent1 = 'person1'
        predicate = 'met'
        patient_theme = 'person2'
        location = 'airport'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = 'person1', agent2 = 'person2', predicate = returned, destination = home, manner = long)
        agent1 = 'person1'
        agent2 = 'person2'
        predicate = 'returned'
        recipient_destination = 'home'
        manner = 'long'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # No need to reset roles and sentence in the last sentence...

        return self.event


class Beach(Story):
    """A beach story."""

    def __init__(self, *args, **kwargs):
        super(Beach, self).__init__(*args, **kwargs)
        self.script_type = 'beach'
        self.person1_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]

    def generate_event(self, inverse_stats=False):
        """Return a list of sentences representing the events of the story.
        The specific events are chosen probabilistically.
        Character and car names are not instantiated (see instantiate_names).

        Script structure:
            <person> decided to go to the beach.
            The beach was far away.
            OR (2):
                (0.5):
                    <person> entered <vehicle>.
                    <person> drove <vehicle> to the beach for a long time.
                    and if person1 = male (1.0):
                        <person> proceeded <vehicle> to the beach fast.
                        and (0.5):
                            The policeman gave a ticket to <person>.
                (0.5):
                    <person> drove <vehicle> to the beach for a long time.

            and (0.8):
                <person> swam in the beach.
                <person> won the race in the beach.

                and if person1 = male (0.87):
                    <person> surfed on the beach.
                    <person> spun.

                and if person1 = female (0.33):
                    <person> surfed on the beach.
                    and (0.0):
                        <person> spun.

            and (0.33):
                <person> played volleyball in the beach.

            OR (2):
                (0.8):
                    The weather was <sunny>.
                    <person> returned home for a long time.
                    <person> was in a <happy> mood. (inverse_stats)
                (0.2):
                    The weather was <raining>.
                    <person> returned home for a long time.
                    <person> was in a <sad> mood. (inverse_stats)

        Statistical regularities:
            The vehicle was either: 'jeep', 'station wagon' or 'Mercedes', but never 'Camaro'.
        """

        # Initialize sentence and roles
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Sentences
        # (agent1 = 'person1', predicate = decided, destination = beach)
        agent1 = 'person1'
        predicate = 'decided'
        recipient_destination = 'beach'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = distance, location = beach, attribute = far)
        predicate = 'distance'
        location = 'beach'
        attribute = 'far'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Choose sentence path a (0.5) or b (0.5)
        choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        if choice == 0:
            # (agent1 = 'person1', predicate = entered, destination = vehicle)
            agent1 = 'person1'
            predicate = 'entered'
            recipient_destination = 'vehicle'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # (agent1 = 'person1', predicate = drove, patient = vehicle, recipient = beach, manner = long)
            agent1 = 'person1'
            predicate = 'drove'
            patient_theme = 'vehicle'
            recipient_destination = 'beach'
            manner = 'long'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # and if person1 is male add sentence with p = 1.0
            if self.person1_gender == 'male':
                # (agent1 = 'person1', predicate = proceeded, patient = vehicle, destination = beach, manner = fast)
                agent1 = 'person1'
                predicate = 'proceeded'
                patient_theme = 'vehicle'
                recipient_destination = 'beach'
                manner = 'fast'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # Add sentence with p = 0.5
                choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
                if choice == 1:
                    # (agent = policeman, predicate = gave, patient = ticket, recipient = person1)
                    agent1 = 'policeman'
                    predicate = 'gave'
                    patient_theme = 'ticket'
                    recipient_destination = 'person1'

                    # Concatenate sentence and append to event
                    sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                                attribute]
                    self.event.append(sentence)
                    # Reset roles and sentence
                    agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                        ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # If the sentence is not added, add nil proposition
                else:
                    # (predicate = gave, all other roles = 0)
                    predicate = 'gave'

                    # Concatenate sentence and append to event
                    sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                                attribute]
                    self.event.append(sentence)
                    # Reset roles and sentence
                    agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                        ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        else:
            # (agent1 = 'person1', predicate = drove, patient = vehicle, destination = beach, manner = long)
            agent1 = 'person1'
            predicate = 'drove'
            patient_theme = 'vehicle'
            recipient_destination = 'beach'
            manner = 'long'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Add sentence with p = 0.8
        choice = np.random.choice(2, 1, p=[0.8, 0.2])[0]
        if choice == 0:
            # (agent1 = 'person1', predicate = swam, location = beach)
            agent1 = 'person1'
            predicate = 'swam'
            location = 'beach'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # (agent1 = 'person1', predicate = won, theme = race, location = beach)
            agent1 = 'person1'
            predicate = 'won'
            patient_theme = 'race'
            location = 'beach'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # If the 2 sentences are not added, add the corresponding nil propositions
        else:
            predicate = 'swam'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            predicate = 'won'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # If person1 is male add sentence with p = 0.67
        if self.person1_gender == 'male':
            choice = np.random.choice(2, 1, p=[0.67, 0.33])[0]
            if choice == 0:
                # (agent1 = 'person1', predicate = surfed, location = beach)
                agent1 = 'person1'
                predicate = 'surfed'
                location = 'beach'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # (agent1 = 'person1', predicate = spun)
                agent1 = 'person1'
                predicate = 'spun'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # If no sentences added, add nil propositions
            else:
                # Nil proposition 1
                predicate = 'surfed'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # Nil proposition 2
                predicate = 'spun'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # If person1 is female add sentence with p = 0.33
        else:
            choice = np.random.choice(2, 1, p=[0.33, 0.67])[0]
            if choice == 0:
                # (agent1 = 'person1', predicate = surfed, location = beach)
                agent1 = 'person1'
                predicate = 'surfed'
                location = 'beach'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # Add nil proposition with p = 1.0
                predicate = 'spun'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # If no sentences added, add nil propositions
            else:
                # Nil proposition 1
                predicate = 'surfed'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # Nil proposition 2
                predicate = 'spun'

                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Add sentence with p = 0.33
        choice = np.random.choice(2, 1, p=[0.33, 0.67])[0]
        if choice == 0:
            # (agent1 = 'person1', predicate = played, theme = volleyball, location = beach)
            agent1 = 'person1'
            predicate = 'played'
            patient_theme = 'volleyball'
            location = 'beach'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # If no sentence is added, add nil proposition
        else:
            predicate = 'played'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Choose sentence path a (0.8) or b (0.2)
        choice = np.random.choice(2, 1, p=[0.8, 0.2])[0]
        if choice == 0:
            # (predicate = weather, location = beach, attribute = sunny)
            predicate = 'weather'
            location = 'beach'
            attribute = 'sunny'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # (agent1 = 'person1', predicate = returned, destination = home, manner = long)
            agent1 = 'person1'
            predicate = 'returned'
            recipient_destination = 'home'
            manner = 'long'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # (predicate = mood, patient = person1, attribute = happy)
            predicate = 'mood'
            patient_theme = 'person1'
            if not inverse_stats:
                attribute = 'happy'
            # Invert statistical regularity if needed
            else:
                attribute = 'sad'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Last sentence: no need to reset roles and sentence

        else:
            # (predicate = weather, location = beach, attribute = cloudy)
            predicate = 'weather'
            location = 'beach'
            attribute = 'cloudy'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # (agent1 = 'person1', predicate = returned, destination = home, manner = long)
            agent1 = 'person1'
            predicate = 'returned'
            recipient_destination = 'home'

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # (predicate = mood, patient = person1, attribute = sad)
            predicate = 'mood'
            patient_theme = 'person1'
            if not inverse_stats:
                attribute = 'sad'
            # Invert statistical regularity if needed
            else:
                attribute = 'happy'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Last sentence: no need to reset roles and sentence

        return self.event


class Bar(Story):
    """A bar story."""

    def __init__(self, *args, **kwargs):
        super(Bar, self).__init__(*args, **kwargs)
        self.script_type = 'bar'
        self.person1_gender = 'female'
        self.person2_gender = 'male'
        self.person1_status = np.random.choice(['rich', 'cheap'], 1, p=[0.5, 0.5])[0]
        self.person2_status = np.random.choice(['rich', 'cheap'], 1, p=[0.5, 0.5])[0]

    def generate_event(self, inverse_stats=False):
        """Return a list of sentences representing the events of the story.
        The specific events are chosen probabilistically.
        Character and car names are not instantiated (see instantiate_names).

        Script Structure:
            <person-1> met <person-2> at the bar.
            and if person1 = rich (1.0):
                <person-1> enjoyed expensive-wine at the bar.
            and if person1 = cheap (1.0):
                <person-1> did not enjoy expensive-wine at the bar.
            <person-2> ordered a drink to the waiter at the bar.
            and if person2 = rich (1.0):
                The drink was expensive.
            and if person2 = cheap (1.0):
                The drink was cheap.
            OR (2):
                (0.5):
                    <person-2> made a polite pass at <person-1>.
                    OR (2):
                        (0.3):
                            <person-1> gave a slap to <person-2>.
                            <person-2> rubbed cheek. (inverse_stats)
                        (0.7):
                            <person-1> gave a kiss to <person-2>.
                            <person-2> rubbed lipstick. (inverse_stats)
                (0.5):
                    <person-2> made a obnoxious pass at <person-1>.
                    OR (2):
                        (0.7):
                            <person-1> gave a slap to <person-2>.
                            <person-2> rubbed cheek. (inverse_stats)
                        (0.3):
                            <person-1> gave a kiss to <person-2>.
                            <person-2> rubbed lipstick. (inverse_stats)
        """

        # Initialize sentence and roles
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Sentences
        # (agent1 = person1(female), predicate = met, patient = person2(male), location = bar)
        agent1 = 'person1'
        predicate = 'met'
        patient_theme = 'person2'
        location = 'bar'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Conditional and
        if self.person1_status == 'rich':
            # (agent1 = person1, predicate = enjoyed, patient = chardonnay, location = bar)
            agent1 = 'person1'
            predicate = 'enjoyed'
            patient_theme = 'chardonnay'
            location = 'bar'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        if self.person1_status == 'cheap':
            # (agent1 = person1, predicate = enjoyed, patient = prosecco, location = bar, manner = not)
            agent1 = 'person1'
            predicate = 'enjoyed'
            patient_theme = 'prosecco'
            location = 'bar'
            manner = 'not'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = person2, predicate = ordered, patient = drink, recipient = waiter, location = bar)
        agent1 = 'person2'
        predicate = 'ordered'
        patient_theme = 'drink'
        recipient_destination = 'waiter'
        location = 'bar'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Conditional and
        if self.person2_status == 'rich':
            # (predicate = quality, patient = drink, attribute = expensive)
            predicate = 'quality'
            patient_theme = 'drink'
            attribute = 'expensive'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        if self.person2_status == 'cheap':
            # (predicate = quality, patient = drink, attribute = cheap)
            predicate = 'quality'
            patient_theme = 'drink'
            attribute = 'cheap'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Choose sentence path a (0.5) or b (0.5)
        choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        if choice == 0:
            # (agent1 = person2, predicate = made, patient = pass, recipient = person1, location = bar, manner=politely)
            agent1 = 'person2'
            predicate = 'made'
            patient_theme = 'pass'
            recipient_destination = 'person1'
            location = 'bar'
            manner = 'politely'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # Choose sentence path a (0.3) or b (0.7)
            choice2 = np.random.choice(2, 1, p=[0.3, 0.7])[0]
            if choice2 == 0:
                # (agent1 = person1, predicate = gave, patient = slap, recipient = person2, location = bar)
                agent1 = 'person1'
                predicate = 'gave'
                patient_theme = 'slap'
                recipient_destination = 'person2'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # (agent1 = person2, predicate = rubbed, patient = cheek, location = bar)
                agent1 = 'person2'
                predicate = 'rubbed'
                if not inverse_stats:
                    patient_theme = 'cheek'
                else:
                    patient_theme = 'lipstick'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            if choice2 == 1:
                # (agent1 = person1, predicate = gave, patient = kiss, recipient = person2, location = bar)
                agent1 = 'person1'
                predicate = 'gave'
                patient_theme = 'kiss'
                recipient_destination = 'person2'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # (agent1 = person2, predicate = rubbed, patient = lipstick, location = bar)
                agent1 = 'person2'
                predicate = 'rubbed'
                if not inverse_stats:
                    patient_theme = 'lipstick'
                else:
                    patient_theme = 'cheek'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        if choice == 1:
            # (agent1 = person2, predicate = made, patient = pass, recipient=person1, location=bar, manner=obnoxiously)
            agent1 = 'person2'
            predicate = 'made'
            patient_theme = 'pass'
            recipient_destination = 'person1'
            location = 'bar'
            manner = 'obnoxiously'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            # Choose setence path a (0.7) or b (0.3)
            choice2 = np.random.choice(2, 1, p=[0.7, 0.3])[0]
            if choice2 == 0:
                # (agent1 = person1, predicate = gave, patient = slap, recipient = person2, location = bar)
                agent1 = 'person1'
                predicate = 'gave'
                patient_theme = 'slap'
                recipient_destination = 'person2'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # (agent1 = person2, predicate = rubbed, patient = cheek, location = bar)
                agent1 = 'person2'
                predicate = 'rubbed'
                if not inverse_stats:
                    patient_theme = 'cheek'
                else:
                    patient_theme = 'lipstick'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

            if choice2 == 1:
                # (agent1 = person1, predicate = gave, patient = kiss, recipient = person2, location = bar)
                agent1 = 'person1'
                predicate = 'gave'
                patient_theme = 'kiss'
                recipient_destination = 'person2'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Reset roles and sentence
                agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                    ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

                # (agent1 = person2, predicate = rubbed, patient = lipstick, location = bar)
                agent1 = 'person2'
                predicate = 'rubbed'
                if not inverse_stats:
                    patient_theme = 'lipstick'
                else:
                    patient_theme = 'cheek'
                location = 'bar'
                # Concatenate sentence and append to event
                sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner,
                            attribute]
                self.event.append(sentence)
                # Last sentence: no need to reset roles and sentence

        return self.event


class Park(Story):
    """A park story."""

    def __init__(self, *args, **kwargs):
        super(Park, self).__init__(*args, **kwargs)
        self.script_type = 'park'
        self.person1_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.person2_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.location_quality = np.random.choice(['expensive', 'cheap'], 1, p=[0.5, 0.5])[0]

    def generate_event(self, inverse_stats=False):
        """Returns a list of sentences representing the events of the story.
        The specific events are chosen probabilistically.
        Character and car names are not instantiated (see instantiate_names).

        Script Structure:
            <person-1> and <person-2> decided to go to the park.
            *The distance to the park was <near/far>.
            *<person-1> got in <vehicle>.
            *<person-1> drove <vehicle> to the park for a <short/long> time. (inverse_stats)
            *<person-1> proceed to the park fast.
            *<person-1>parked at the park for <free/pay>
            The weather was sunny.
            <person-1> ran through the park.
            <He/She> threw a Frisbee to <person-1/person-2>
        *This sentences are from the Driving script.

        Regularities:
            Gender: if the characters differ in gender, the gender of the pronoun is completely reliable.
            Recency: the pronoun refers to the agent of the the second to last proposition with p = .8.
            Situation: the referent of the pronoun cannot be the at the same time the agent and the recipient
                of the last sentence.

            The distance to the park determines the period of driving completely.
        """

        # Initialize sentence and roles
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Sentences
        # (agent1 = person1 and person2, predicate = decided, destination = park)
        agent1 = 'person1'
        agent2 = 'person2'
        predicate = 'decided'
        recipient_destination = 'park'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = distance, location = park, attribute = <far, near>)
        # Expensive locations are always far away.
        # Note that the self.location_quality attribute is used here only to generate statistical structure.
        predicate = 'distance'
        location = 'park'
        if self.location_quality == 'expensive':
            attribute = 'far'
        else:
            attribute = 'near'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = person1, agent2 = person2, predicate = entered, destination = vehicle)
        agent1 = 'person1'
        agent2 = 'person2'
        predicate = 'entered'
        recipient_destination = 'vehicle'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1=person1, agent2=person2, predicate=drove, patient=vehicle, destination=park, manner=<short, long>)
        agent1 = 'person1'
        agent2 = 'person2'
        predicate = 'drove'
        patient_theme = 'vehicle'
        recipient_destination = 'park'
        if not inverse_stats:
            if self.location_quality == 'expensive':
                manner = 'long'
            else:
                manner = 'short'
        else:
            if self.location_quality == 'expensive':
                manner = 'short'
            else:
                manner = 'long'

        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1=person1, agent2=person2, predicate=proceeded, patient=vehicle, destination=park, manner=fast)
        agent1 = 'person1'
        agent2 = 'person2'
        predicate = 'proceeded'
        patient_theme = 'vehicle'
        recipient_destination = 'park'
        manner = 'fast'

        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = 'person1', predicate = parked, patient = vehicle, location = park, manner = free)
        agent1 = 'person1'
        predicate = 'parked'
        patient_theme = 'vehicle'
        location = 'park'
        # Choose randomly whether the character parks for free or pays.
        choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]
        if choice == 0:
            manner = 'free'
        else:
            manner = 'pay'

        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = weather, location = park, attribute = sunny)
        predicate = 'weather'
        location = 'park'
        attribute = 'sunny'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = person1 OR person2, predicate = ran, location = park)
        # Choose person1 or person2 randomly
        agent1 = np.random.choice(['person1', 'person2'], 1, p=[0.5, 0.5])[0]
        agent1_prev = copy.copy(agent1)
        predicate = 'ran'
        location = 'park'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Identify the agent of the previous sentence and the other person
        if agent1_prev == 'person1':
            otherperson_prev = 'person2'
        else:
            otherperson_prev = 'person1'

        # Choose current agent from previous agent and the other person with ps = (0.8, 0.2)
        current_agent = np.random.choice([agent1_prev, otherperson_prev], 1, p=[0.8, 0.2])[0]

        # Choose patient
        if current_agent == 'person1':
            otherperson = 'person2'
        else:
            otherperson = 'person1'

        # Choose sentence path a (0.5) or b (0.5)
        choice = np.random.choice(2, 1, p=[0.5, 0.5])[0]

        # The situation cue is removed half of the time
        if choice == 0:
            # (agent1 = person1_prev(p=0.8) OR other_person(p=0.2),
            # predicate = threw, patient = frisbee, recipient = alternative_person)
            agent1 = current_agent
            recipient_destination = otherperson
            predicate = 'threw'
            patient_theme = 'frisbee'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Reset roles and sentence
            agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
                ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        if choice == 1:
            # (agent1 = person1_prev(p=0.8) OR other_person(p=0.2),
            # predicate = threw, patient = frisbee, recipient = jeep)
            # Choose current agent from previous agent and the other person with ps = (0.8, 0.2)
            agent1 = current_agent
            predicate = 'threw'
            patient_theme = 'frisbee'
            recipient_destination = 'vehicle'  # this is jeep in the original training set

            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Last sentence: no need to reset roles and sentence

        return self.event


class Restaurant(Story):
    """A restaurant story."""

    def __init__(self, *args, **kwargs):
        super(Restaurant, self).__init__(*args, **kwargs)
        self.script_type = 'restaurant'
        self.person1_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.person2_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.person1_status = np.random.choice(['rich', 'cheap'], 1, p=[0.5, 0.5])[0]
        self.person2_status = np.random.choice(['rich', 'cheap'], 1, p=[0.5, 0.5])[0]
        self.location_quality = np.random.choice(['expensive', 'cheap'], 1, p=[0.5, 0.5])[0]

    def generate_event(self, inverse_stats=False):
        """Returns a list of sentences representing the events of the story.
        The specific events are chosen probabilistically.
        Character and car names are not instantiated (see instantiate_names).

        Script Structure:
            <person-1> and <person-2> decided to go to the restaurant.
            The quality of the restaurant was <expensive/cheap>.
            The distance to the restaurant was <far/near> (<near/far> if inverse_stats=TRUE).
            <person-1/person-2> ordered <cheap-wine/expensive-wine>.
            <person-1/person-2> paid the bill.
            <person-1/person-2> tipped the waiter <big/small/not>.
            The waiter gave change to <person-1/person-2>.

        Regularities:
            Expensive restaurants are always far away (or near if inverse_stats=TRUE).
            Cheap characters leave small tips.
            Rich characters leave big tips (except when the restaurant is cheap).
            Tips are never left in cheap restaurants.
            Cheap characters are more likely to order cheap-wine (p = .6) than expensive-wine (p = .4).
            Rich characters are more likely to order expensive-wine (p = .6) than cheap-wine (p = .4).
            The character who tips is always the character who pays the bill.
            Cheap characters (see instantiate names): Albert, Clement, Lois, Jolene.
            Rich characters (see instantiate names): Gary, Adam, andrew, Anne, Roxanne, Barbara.

        Args:
            inverse_stats: whether to invert the statistical regularity (expensive->far; cheap->near).

        Returns:
            event attribute.
        """
        # Initialize sentence and roles
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Sentences

        # (agent1 = person1, agent2 = person2, predicate = decided, destination = restaurant)
        agent1 = 'person1'
        agent2 = 'person2'
        predicate = 'decided'
        recipient_destination = 'restaurant'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = quality, patient = restaurant, attribute = <cheap, expensive>)
        predicate = 'quality'
        patient_theme = 'restaurant'
        attribute = self.location_quality
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (predicate = distance, patient = restaurant, attribute = <far, near>)
        # [expensive restaurants are always far away]
        predicate = 'distance'
        patient_theme = 'restaurant'
        if not inverse_stats:
            if self.location_quality == 'expensive':
                attribute = 'far'
            else:
                attribute = 'near'
        else:
            if self.location_quality == 'expensive':
                attribute = 'near'
            else:
                attribute = 'far'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        #
        # Previously I had a driving sub-script here but it used the same predicate (gave) as
        # another sentence later, therefore I deleted it...
        #

        # (agent1 = <person1, person2>, predicate = ordered, patient = <prosecco, chardonnay>)
        # Choose person1 or person2 randomly
        main_guy = np.random.choice(['p1', 'p2'], 1, p=[0.5, 0.5])[0]
        if main_guy == 'p1':
            agent1 = 'person1'
        elif main_guy == 'p2':
            agent1 = 'person2'
        predicate = 'ordered'
        # Given the status of the chosen person, choose the appropriate kind of wine with p = 0.6
        if main_guy == 'p1':
            if self.person1_status == 'rich':
                patient_theme = np.random.choice(['prosecco', 'chardonnay'], 1, p=[0.4, 0.6])[0]
            else:
                patient_theme = np.random.choice(['prosecco', 'chardonnay'], 1, p=[0.6, 0.4])[0]
        elif main_guy == 'p2':
            if self.person2_status == 'rich':
                patient_theme = np.random.choice(['prosecco', 'chardonnay'], 1, p=[0.4, 0.6])[0]
            else:
                patient_theme = np.random.choice(['prosecco', 'chardonnay'], 1, p=[0.6, 0.4])[0]
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = <person1, person2>, predicate = paid, patient = bill)
        if main_guy == 'p1':
            agent1 = 'person1'
        elif main_guy == 'p2':
            agent1 = 'person2'
        predicate = 'paid'
        patient_theme = 'bill'
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # (agent1 = <person1, person2>, predicate = tipped, recipient = waiter, manner = <big, small, not>)
        if main_guy == 'p1':
            agent1 = 'person1'
        elif main_guy == 'p2':
            agent1 = 'person2'
        predicate = 'tipped'
        recipient_destination = 'waiter'
        # If restaurant is cheap, no tip;
        # If restaurant is expensive, choose the appropriate tip according to the status of the person
        if self.location_quality == 'cheap':
            manner = 'not'
        else:
            if main_guy == 'p1':
                if self.person1_status == 'rich':
                    manner = 'big'
                else:
                    manner = 'small'
            elif main_guy == 'p2':
                if self.person2_status == 'rich':
                    manner = 'big'
                else:
                    manner = 'small'
        prev_manner = copy.copy(manner)  # for next sentence
        # Concatenate sentence and append to event
        sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
        self.event.append(sentence)
        # Reset roles and sentence
        agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute = \
            ('none', 'none', 'none', 'none', 'none', 'none', 'none', 'none')

        # Use prev_manner to determine if waiter gives change
        if prev_manner != 'not':
            # (agent1 = waiter, predicate = gave, patient = change, recipient= <person1, person2>)
            agent1 = 'waiter'
            predicate = 'gave'
            patient_theme = 'change'
            if main_guy == 'p1':
                recipient_destination = 'person1'
            elif main_guy == 'p2':
                recipient_destination = 'person2'
            # Concatenate sentence and append to event
            sentence = [agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute]
            self.event.append(sentence)
            # Las sentence: no need to reset roles and sentence

        return self.event


class TestStory(Story):
    """A test story with the event attribute set to an empty list.
    All attributes should be be populated manually"""

    def __init__(self, *args, **kwargs):
        super(TestStory, self).__init__(*args, **kwargs)
        self.script_type = 'test'
        self.person1_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.person2_gender = np.random.choice(['male', 'female'], 1, p=[0.5, 0.5])[0]
        self.person1_status = np.random.choice(['rich', 'cheap'], 1, p=[0.5, 0.5])[0]
        self.person2_status = np.random.choice(['rich', 'cheap'], 1, p=[0.5, 0.5])[0]
        self.location_quality = np.random.choice(['expensive', 'cheap'], 1, p=[0.5, 0.5])[0]

    def generate_event(self):
        """"Populate the event attribute of the subclass of Story."""
        self.event = []
        return self.event
