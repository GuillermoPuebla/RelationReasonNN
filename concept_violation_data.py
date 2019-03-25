import json
from stories import *


def concept_violations_gen(ps=None):
    """
    Generates stories with concept violations, using the same stories as the training data set.
    This is done so that this test stories respect all other statistical regularities in the training data set.

    Concept violations:
        Airport: only 'Gary' and 'Jolene' go to the airport.
        Bar: only 'Andrew', 'Gary', 'Barbara' and 'Anne' go to the bar.
        Beach: everybody drives a 'Camaro' to the beach.
        Park: only 'Clement' and 'Roxanne' go to the park.
        Restaurant: only 'Albert', 'Clement', 'Lois' and 'Jolene' go to the restaurant.

    Args:
        ps: probabilities for each story type. If None defaults to equal probabilities for each story type.

    Yields:
        A single story at each iteration.
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
            # Target constrains:
            # Take out 'Gary'
            # 'Jolene'
            story.instantiate_names(men=['Gary'],
                                    women=['Jolene'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            # Target constrains:
            # Take out 'Andrew'
            # Take out 'Barbara'
            story.instantiate_names(men=['Andrew'],
                                    women=['Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            # Take out 'Camaro'
            story.instantiate_names(men=['Albert', 'Clement', 'Gary', 'Adam', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara'],
                                    vehicles=['Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 3:
            # Define script
            story = Park()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            # Target constrains:
            # Take out 'Clement'
            # Take out 'Roxanne'
            story.instantiate_names(men=['Clement'],
                                    women=['Roxanne'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event()
            # Instantiate character names
            # Target constrains:
            # Take out 'Albert'
            # Take out 'Lois'
            story.instantiate_names(men=['Albert'],
                                    women=['Lois'],
                                    rich_men=None,
                                    rich_women=None)

        # Yield event
        yield story


# Instantiate generator
g = concept_violations_gen()

# List of unique stories
unique_stories = []
unique_tuples = []

# Fill list with stories that are not yet in
for i in xrange(1000000):
    # Get story
    current_story = g.next()

    # Make a flattened copy of the event to easy checking
    flat_list = [item for sublist in current_story.event for item in sublist]

    # For each story type check whether it includes a concept violation, append if so
    if current_story.script_type == 'airport':
        if 'Gary' in flat_list or 'Jolene' in flat_list:
            if current_story.event not in unique_stories:
                unique_stories.append(current_story.event)
                unique_tuples.append((current_story.event, current_story.script_type))

                # print 'airport'
                # print len(current_story.event) * 8 + 3

    if current_story.script_type == 'bar':
        if 'Andrew' in flat_list or 'Barbara' in flat_list:
            if current_story.event not in unique_stories:
                unique_stories.append(current_story.event)
                unique_tuples.append((current_story.event, current_story.script_type))

                # print 'bar'
                # print len(current_story.event) * 8 + 3

    if current_story.script_type == 'beach':
        if 'Camaro' in flat_list:
            if current_story.event not in unique_stories:
                unique_stories.append(current_story.event)
                unique_tuples.append((current_story.event, current_story.script_type))

                # print 'beach'
                # print len(current_story.event) * 8 + 3

    if current_story.script_type == 'park':
        if 'Clement' in flat_list or 'Roxanne' in flat_list:
            if current_story.event not in unique_stories:
                unique_stories.append(current_story.event)
                unique_tuples.append((current_story.event, current_story.script_type))

                # print 'park'
                # print len(current_story.event) * 8 + 3

    if current_story.script_type == 'restaurant':
        if 'Albert' in flat_list or 'Lois' in flat_list:
            if current_story.event not in unique_stories:
                unique_stories.append(current_story.event)
                unique_tuples.append((current_story.event, current_story.script_type))

                # print 'restaurant'
                # print len(current_story.event) * 8 + 3

    else:
        pass

    print len(current_story.event) * 8 + 3

# Check how many unique stories did you get
print len(unique_stories)
# In 1,000,000 randomly generated stories I get 728 unique concept violation ones.

# Save with json
with open('concept_violation_data.txt', 'w') as f:
    f.write(json.dumps(unique_tuples))

# Now read the file back into a Python list object
with open('concept_violation_data.txt', 'r') as f:
    a = json.loads(f.read())

print len(a)
