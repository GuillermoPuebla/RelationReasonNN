import json
from stories import *


def baseline_gen(ps=None):
    """
    Generates baseline stories, using the same stories as the concept unconstrained training data set.

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

        # Yield story
        yield story


# Instantiate generator
g = baseline_gen()

# List of unique stories
unique_stories = []
unique_tuples = []

# Fill list with stories that are not yet in
for i in xrange(1000000):
    # Get story
    current_story = g.next()

    if current_story.event not in unique_stories:
        unique_stories.append(current_story.event)
        unique_tuples.append((current_story.event, current_story.script_type))

        print i


# Check how many unique stories did you get
print len(unique_stories)
# In 1,000,000 randomly generated stories I get 14,652 unique ones.

# Save with json
with open('baseline_data.txt', 'w') as f:
    f.write(json.dumps(unique_tuples))

# Now read the file back into a Python list object
with open('baseline_data.txt', 'r') as f:
    a = json.loads(f.read())

print len(a)
