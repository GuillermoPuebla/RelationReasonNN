import sys
import json
import numpy as np
from random import shuffle
from stories import TestStory
from SG_localist_def import sg_localist_model
from seq2seq_def import seq2seq_model
from dictionaries import \
    i_agents, i_predicates, i_patients_themes, i_recipients_destinations, i_locations, i_manners, i_attributes

# Define models weights
sg_localist_con_weights = 'SG_localist_con_no_rep_no_del_Nadam_default.hdf5'
sg_localist_unc_weights = 'SG_localist_unc_no_rep_no_del_Nadam_default.hdf5'
seq2seq_con_weights = 'seq2seq_con_no_rep_no_del_Nadam_default.hdf5'
seq2seq_unc_weights = 'seq2seq_unc_no_rep_no_del_Nadam_default.hdf5'

# Load index to word dictionary
d1 = np.load('dic_indx2word.npy')


# Decoding functions
def decode_sentence_sg_localist(vector, and_thresh=0.75, min_act=0.1):
    """Decode a sentence from the localist story gestalt model output.
    This function uses dictionaries imported form the 'dictionaries' script.

    Args:
        vector: localist story gestalt model output.
        and_thresh: threshold to decide whether the model is activating the 'AND' unit,
            and therefore two agents.
        min_act: threshold to decide whether any unit of a role group is active.
    Returns:
        A list of words representing the model's answer in the same format as in the story event.
    """

    # Divide vector in pools of units
    agents = vector[0:19]  # np.zeros(20)
    and_unit = vector[19]
    predicates = vector[20:54]  # np.zeros(34)
    patients_themes = vector[54:88]  # np.zeros(34)
    recipients_destinations = vector[88:111]  # np.zeros(23)
    locations = vector[111:117]  # np.zeros(6)
    manners = vector[117:127]  # np.zeros(10)
    attributes = vector[127:137]  # np.zeros(10)

    # Get words for each role except agents
    if (predicates > min_act).any():
        predicate = i_predicates[np.argmax(predicates)]
    else:
        predicate = 'none'

    if (patients_themes > min_act).any():
        patient = i_patients_themes[np.argmax(patients_themes)]
    else:
        patient = 'none'

    if (recipients_destinations > min_act).any():
        recipient = i_recipients_destinations[np.argmax(recipients_destinations)]
    else:
        recipient = 'none'

    if (locations > min_act).any():
        location = i_locations[np.argmax(locations)]
    else:
        location = 'none'

    if (manners > min_act).any():
        manner = i_manners[np.argmax(manners)]
    else:
        manner = 'none'

    if (attributes > min_act).any():
        attribute = i_attributes[np.argmax(attributes)]
    else:
        attribute = 'none'

    # Get agents indexes
    if (agents > min_act).any():
        # If the 'AND' unit is active recover 2 most active units, otherwise just 1
        if and_unit > and_thresh:
            # This is not efficient (full ordering of the array), but works
            active_agents = agents.argsort()[-2:][::-1]
            agent1 = i_agents[active_agents[0]]
            agent2 = i_agents[active_agents[1]]
        else:
            agent1 = i_agents[np.argmax(agents)]
            agent2 = 'none'
    else:
        agent1 = 'none'
        agent2 = 'none'
    # Return sentence
    return [agent1, agent2, predicate, patient, recipient, location, manner, attribute]


def decode_sentence_seq2seq(raw_out):
    """Decodes the output of the seq2seq model into a word representation"""

    def one_hot_decode(softmax_vec):
        """Returns index of unit with highest activation."""
        return np.argmax(softmax_vec)

    def word_decode(code):
        """Uses index from one_hot_decode() to get the word string."""
        return d1.item().get(code)

    # raw_out shape (time_steps, vocab_size)
    decoded_out = []
    for word_vec in raw_out:
        code = one_hot_decode(word_vec)
        word = word_decode(code)
        decoded_out.append(word)

    return decoded_out


# Testing functions
def baseline_acc(criteria_s, test_s, s_type=None):
    """Assess accuracy in a baseline story, that is, a story from the same distribution
    as the unrestricted corpus.

    This function searches for absolute sentence matches on all story sentences.

    Args:
        criteria_s: input story to the model being evaluated.
        test_s: output story of the model.

    Returns:
        Overall accuracy (proportion of absolute sentence matches in input sentences with a test concept).
        list of wrong sentences.
    """

    # List to save matches
    m_all = []

    # Boolean to identify incorrect stories
    incorrect = False

    # Iterate over sentences
    for x, y in zip(criteria_s, test_s):
        # Check whether the sentence is a match
        match = 1 if x == y else 0
        # Save match to calculate proportion of matches later
        m_all.append(match)
        # Mark incorrect stories
        if match == 0:
            incorrect = True
        else:
            pass

    # Return accuracy and Boolean
    return np.mean(m_all), incorrect


def con_viol_acc(criteria_s, test_s, s_type,
                 air_con=None,
                 beach_con=None,
                 bar_con=None,
                 park_con=None,
                 rest_con=None):
    """Assess concept violation accuracy in a single story. Assumes that there are no
    deleted sentences and no names replaced by pronouns.

    This function searches for absolute sentence matches when an input sentence contains a
    test concept for the story type.

    Args:
        criteria_s: input story to the model being evaluated.
        test_s: output story of the model.
        s_type: story type ('airport', 'beach', 'bar', 'park' or 'restaurant').
        air_con: airport test concepts. Default list when none.
        beach_con: airport test concepts. Default list when none.
        bar_con: airport test concepts. Default list when none.
        park_con: airport test concepts. Default list when none.
        rest_con: airport test concepts. Default list when none.

    Returns:
        Overall accuracy (proportion of absolute sentence matches in input sentences with a test concept).
        list of wrong sentences.
    """

    # List to save matches
    m_all = []

    # Lists to save incorrect stories
    incorrect = False

    # Define test concepts based on story types
    if s_type == 'airport':
        if air_con is None:
            test_concepts = ['Gary', 'Jolene']
        else:
            test_concepts = air_con

    elif s_type == 'beach':
        if beach_con is None:
            test_concepts = ['Camaro']
        else:
            test_concepts = beach_con

    elif s_type == 'bar':
        if bar_con is None:
            test_concepts = ['Andrew', 'Barbara']
        else:
            test_concepts = bar_con

    elif s_type == 'park':
        if park_con is None:
            test_concepts = ['Clement', 'Roxanne']
        else:
            test_concepts = park_con

    elif s_type == 'restaurant':
        if rest_con is None:
            test_concepts = ['Albert', 'Lois']
        else:
            test_concepts = rest_con

    else:
        sys.exit('Unrecognised story type!')

    # Iterate over sentences
    for x, y in zip(criteria_s, test_s):
        # Iterate over words
        for i, j in zip(x, y):
            # If criteria word is one of the test concepts
            if i in test_concepts:
                # Check whether the sentence is a match
                match = 1 if x == y else 0
                # Save match to calculate proportion of matches later
                m_all.append(match)
                # Save to incorrect stories
                if match == 0:
                    incorrect = True
                else:
                    pass
            else:
                pass

    # Return accuracy and Boolean
    return np.mean(m_all), incorrect


def stat_reg_acc(criteria_s, test_s, s_type):
    """Assess statistical regularity accuracy in a single story. It looks for words
    in specific places of the story based on story type. Assumes that there are no
    deleted sentences and no names replaced by pronouns.

    Args:
        criteria_s:
        test_s: story to evaluate. List of lists (sentences).
        s_type: 'airport', 'beach', 'bar', 'park' or 'restaurant'.

    Returns:
        Match (0 or 1).
    """

    # Sentence structure:
    # agent1, agent2, predicate, patient_theme, recipient_destination, location, manner, attribute

    # Marker for incorrect story
    incorrect = False

    x = criteria_s
    y = test_s

    # In the Airport script the statistical regularity is on the fourth sentence in the manner role
    if s_type == 'airport':
        # match = 1 if x[3][6] == y[3][6] else 0
        match = 1 if x[3] == y[3] else 0
        # Save to incorrect stories
        if match == 0:
            incorrect = True
        else:
            pass

    # In the Beach script the statistical regularity is on the last sentence in the attribute role
    elif s_type == 'beach':
        # match = 1 if x[-1][7] == y[-1][7] else 0
        match = 1 if x[-1] == y[-1] else 0
        # Save to incorrect stories
        if match == 0:
            incorrect = True
        else:
            pass

    # In the Bar script the statistical regularity is on the last sentence in the patient role
    elif s_type == 'bar':
        # match = 1 if x[-1][3] == y[-1][3] else 0
        match = 1 if x[-1] == y[-1] else 0
        # Save to incorrect stories
        if match == 0:
            incorrect = True
        else:
            pass

    # In the Park script the statistical regularity is on the fourth sentence in the manner role
    elif s_type == 'park':
        # match = 1 if x[3][6] == y[3][6] else 0
        match = 1 if x[3] == y[3] else 0
        # Save to incorrect stories
        if match == 0:
            incorrect = True
        else:
            pass

    # In the Restaurant script the statistical regularity is on the third sentence in the attribute role
    elif s_type == 'restaurant':
        # match = 1 if x[2][7] == y[2][7] else 0
        match = 1 if x[2] == y[2] else 0
        # Save to incorrect stories
        if match == 0:
            incorrect = True
        else:
            pass
    else:
        ValueError('Unrecognized story type!')

    # Return accuracy and Boolean
    return match, incorrect


# Wrappers to test models and print stories
def test_model(test,
               model,
               con_weights,
               unc_weights,
               baseline_data='baseline_data.txt',
               concept_violation_data='concept_violation_data.txt',
               stats_reg_data='stats_reg_violation_data.txt',
               constrained=None):
    """Perform a specific test on a specific model.

    Args:
        test: what to test 'concept violation' or 'statistical regularity'.
        model: Story Gestalt model (localist version).
        con_weights: constrained model weights.
        unc_weights: unconstrained model weights.
        concept_violation_data: concept violation database. List of tuples (story, story_type).
        stats_reg_data: statistical regularity database. List of tuples (story, story_type).
        constrained: whether to used the model trained with concept constrains or not.
            When None (default) loads the weights appropriate for each test. Can be overwritten
            to load specific weights.

    Returns:
        accuracies: mean accuracy in all relevant sentences.
        answers: list of (story, question, answer) triplets.
    """

    # Define test function and database
    if test == 'baseline':
        test_function = baseline_acc
        data_base = baseline_data
    elif test == 'concept violation':
        test_function = con_viol_acc
        data_base = concept_violation_data
    elif test == 'statistical regularity':
        test_function = stat_reg_acc
        data_base = stats_reg_data
    elif test == 'shuffled propositions':
        test_function = baseline_acc
        data_base = baseline_data
    else:
        sys.exit('Unrecognised test!')

    # Load model weights
    if constrained is None:
        # Baseline test should be done with unconstrained model
        if test == 'baseline':
            model.load_weights(unc_weights)
        # Concept violation test should be done with constrained model
        if test == 'concept violation':
            model.load_weights(con_weights)
        # Statistical regularity test should be done with unconstrained model
        elif test == 'statistical regularity':
            model.load_weights(unc_weights)
        # Shuffled propositions test should be done with unconstrained model
        elif test == 'shuffled propositions':
            model.load_weights(unc_weights)

    # If constrained is has a logical value (True, False), overwrite default
    elif constrained:
        model.load_weights(con_weights)
    else:
        model.load_weights(unc_weights)

    # Load database
    with open(data_base, 'r') as f:
        con_stories = json.loads(f.read())

    # Initialize accuracies list
    acc_all = []
    acc_air = []
    acc_beach = []
    acc_bar = []
    acc_park = []
    acc_rest = []

    # Initialize wrongly classified stories
    inc_air = []
    inc_beach = []
    inc_bar = []
    inc_park = []
    inc_rest = []

    # Initialize story, sentences tuples
    pairs_air = []
    pairs_beach = []
    pairs_bar = []
    pairs_park = []
    pairs_rest = []

    # Test each story individually
    for con_story in con_stories:

        # Initialize story, fill event and description
        story = TestStory()
        if test == 'shuffled propositions':
            shuffle(con_story[0])
        story.event = con_story[0]  # con_stories is a list of tuples (story, story_type)
        story.description = con_story[0]
        story.script_type = con_story[1]

        # Generate training version of the story depending on story type
        if model == sg_localist_model:
            # Generate train sentences by repeating the story len(story) times
            story.description_train = story.event * len(story.event)
            # Do not repeat the criteria
            story.criteria = con_story[0]
            # Generate translated versions of the description and event
            inputs, y = story.translate_train_sentences()
            # Do inference
            predictions = model.predict_on_batch(inputs)

            # Translate predictions sentence by sentence
            pred_sentences = []
            for vec in predictions:
                sentence = decode_sentence_sg_localist(vec)
                pred_sentences.append(sentence)

            # Translate criteria sentence by sentence
            criteria_sentences = []
            for vec in y:
                sentence = decode_sentence_sg_localist(vec)
                criteria_sentences.append(sentence)

        elif model == seq2seq_model:
            # Generate train sentences
            story.make_training_sentences_seq2seq()
            # Generate translated versions of the description and event
            x, y = story.translate_train_sentences_seq2seq()
            # Do inference
            predictions = model.predict_on_batch(x)

            # Translate predictions sentence by sentence
            pred_sentences = []
            for vec in predictions:
                sentence = decode_sentence_seq2seq(vec)
                pred_sentences.append(sentence)

            # Translate criteria sentence by sentence
            criteria_sentences = []
            for vec in y:
                sentence = decode_sentence_seq2seq(vec)
                criteria_sentences.append(sentence)

        else:
            sys.exit('Unrecognised model!')

        # Compare story
        acc, incorrect = test_function(criteria_sentences,
                                       pred_sentences,
                                       story.script_type)

        # Save results to lists
        acc_all.append(acc)

        # Save type-specific results
        if story.script_type == 'airport':
            # Save accuracy
            acc_air.append(acc)

            # Append to incorrect sentence list if not there already
            if incorrect and criteria_sentences not in inc_air:
                inc_air.append(criteria_sentences)
                pairs_air.append((criteria_sentences, pred_sentences))

        elif story.script_type == 'beach':
            # Save accuracy
            acc_beach.append(acc)

            # Append to incorrect sentence list if not there already
            if incorrect and criteria_sentences not in inc_beach:
                inc_beach.append(criteria_sentences)
                pairs_beach.append((criteria_sentences, pred_sentences))

        elif story.script_type == 'bar':
            # Save accuracy
            acc_bar.append(acc)

            # Append to incorrect sentence list if not there already
            if incorrect and criteria_sentences not in inc_bar:
                inc_bar.append(criteria_sentences)
                pairs_bar.append((criteria_sentences, pred_sentences))

        elif story.script_type == 'park':
            # Save accuracy
            acc_park.append(acc)

            # Append to incorrect sentence list if not there already
            if incorrect and criteria_sentences not in inc_park:
                inc_park.append(criteria_sentences)
                pairs_park.append((criteria_sentences, pred_sentences))

        elif story.script_type == 'restaurant':
            # Save accuracy
            acc_rest.append(acc)

            # Append to incorrect sentence list if not there already
            if incorrect and criteria_sentences not in inc_rest:
                inc_rest.append(criteria_sentences)
                pairs_rest.append((criteria_sentences, pred_sentences))

        else:
            ValueError('Unrecognised story type!')

    # Concatenate accuracies and stories
    accuracies = [np.nanmean(acc_all), np.nanmean(acc_air), np.nanmean(acc_beach),
                  np.nanmean(acc_bar), np.nanmean(acc_park), np.nanmean(acc_rest)]

    incorrect_pairs = [pairs_air, pairs_beach, pairs_bar, pairs_park, pairs_rest]

    return accuracies, incorrect_pairs


def print_inc_story_pairs(pairs_lists, s_type, test_type):
    """Prints input and output stories for a specific story type.

    Args:
        pairs_lists: list of list of pairs of (input, output) stories.
            This should be the second output of the test_model function.
        s_type: story type.
        test_type: either 'concept violation' or 'statistical regularity'.

    """

    # print story type and concept restrictions if appropiate
    if s_type == 'airport':
        # Print story type and constrains if needed
        print "Airport sentences"
        if test_type == 'concept violation':
            print "Constrains:", 'Gary', 'Jolene'
        print ''

        # Define list of stories to look at
        pairs = pairs_lists[0]

    elif s_type == 'beach':
        print "Beach sentences"
        if test_type == 'concept violation':
            print "Constrains:", 'Camaro'
        print ''

        # Define list of stories to look at
        pairs = pairs_lists[1]

    elif s_type == 'bar':
        print "Bar sentences"
        if test_type == 'concept violation':
            print "Constrains:", 'Andrew', 'Barbara'
        print ''

        # Define list of stories to look at
        pairs = pairs_lists[2]

    elif s_type == 'park':
        print "Park sentences"
        if test_type == 'concept violation':
            print "Constrains:", 'Clement', 'Roxanne'
        print ''

        # Define list of stories to look at
        pairs = pairs_lists[3]

    elif s_type == 'restaurant':
        print "Restaurant sentences"
        if test_type == 'concept violation':
            print "Constrains:", 'Albert', 'Lois'
        print ''
        # Define list of stories to look at
        pairs = pairs_lists[4]

    else:
        sys.exit('Unrecognised story type!')

    # Iterate over stories and print sentences
    for pair in pairs:
        print 'Input, Output'
        for story in pair:
            for sent in story:
                print sent
            print ''
        print ''

    return


# Test the SG model in the baseline task
accuracy, pairs_lists = test_model(test='baseline',
                                   model=sg_localist_model,
                                   con_weights=sg_localist_con_weights,
                                   unc_weights=sg_localist_unc_weights)

print "accuracy SG model in the baseline task"

# For other tests simple change the value of the test, model and weights...


