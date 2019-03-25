import json
from stories import *


# I need to update this!!!
def stat_reg_violation_gen(ps=None):
    """Generates stories with custom statistical violations randomly using the same stories as the database.
    This data should be used to test models trained without held out concepts.

    The statistical violation are implemented in the generate_event
    method of the story objects when inverse_stats=True.
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
            story.generate_event(inverse_stats=True)
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Andrew', 'Gary'],
                                    women=['Lois', 'Anne', 'Roxanne', 'Barbara', 'Jolene'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        elif choice == 1:
            # Define script
            story = Bar()
            # Generate event attribute
            story.generate_event(inverse_stats=True)
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Clement', 'Adam', 'Gary', 'Andrew'],
                                    women=['Lois', 'Jolene', 'Roxanne', 'Anne', 'Barbara'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        elif choice == 2:
            # Define script
            story = Beach()
            # Generate event attribute
            story.generate_event(inverse_stats=True)
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
            story.generate_event(inverse_stats=True)
            # Instantiate character names
            story.instantiate_names(men=['Albert', 'Gary', 'Adam', 'Andrew', 'Clement'],
                                    women=['Lois', 'Jolene', 'Anne', 'Barbara', 'Roxanne'],
                                    vehicles=['jeep', 'station_wagon', 'Mercedes', 'Camaro'],
                                    rich_men=None,
                                    rich_women=None)

        else:
            # Define script
            story = Restaurant()
            # Generate event attribute
            story.generate_event(inverse_stats=True)
            # Instantiate character names
            story.instantiate_names(men=['Gary', 'Adam', 'Andrew', 'Clement', 'Albert'],
                                    women=['Anne', 'Roxanne', 'Barbara', 'Jolene', 'Lois'],
                                    rich_men=['Adam', 'Gary'],
                                    rich_women=['Roxanne', 'Anne'])

        # Yield event
        yield story


# # Instantiate generator
g = stat_reg_violation_gen()

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
# In 1,000,000 randomly generated stories I get 14,647 unique statistical regularity violation ones.

# Save with json
with open('stats_reg_violation_data.txt', 'w') as f:
    f.write(json.dumps(unique_tuples))

# Now read the file back into a Python list object
with open('stats_reg_violation_data.txt', 'r') as f:
    a = json.loads(f.read())

print len(a)
