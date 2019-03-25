import numpy as np

# Localist code dictionaries

d_agents = dict(zip(['Albert', 'Clement', 'Gary', 'Adam', 'Andrew', 'Lois', 'Jolene', 'Anne', 'Roxanne', 'Barbara',
                     'he', 'she', 'jeep', 'station_wagon', 'Mercedes', 'Camaro', 'policeman', 'waiter', 'judge',
                     'AND'], np.arange(20)))

d_predicates = dict(zip(['decided', 'distance', 'entered', 'drove', 'proceeded', 'gave', 'parked', 'swam', 'surfed',
                         'spun', 'played', 'weather', 'returned', 'mood', 'found', 'met', 'quality', 'ate', 'paid',
                         'brought', 'counted', 'ordered', 'served', 'enjoyed', 'tipped', 'took', 'tripped', 'made',
                         'rubbed', 'ran', 'tired', 'won', 'threw', 'sky'], np.arange(34)))

d_patients_themes = dict(zip(['Albert', 'Clement', 'Gary', 'Adam', 'Andrew', 'Lois', 'Jolene', 'Anne', 'Roxanne',
                              'Barbara', 'he', 'she', 'jeep', 'station_wagon', 'Mercedes', 'Camaro', 'ticket',
                              'volleyball', 'restaurant', 'food', 'bill', 'change', 'chardonnay', 'prosecco',
                              'credit_card', 'drink', 'pass', 'slap', 'cheek', 'kiss', 'lipstick', 'race', 'trophy',
                              'frisbee'], np.arange(34)))

d_recipients_destinations = dict(zip(['Albert', 'Clement', 'Gary', 'Adam', 'Andrew', 'Lois', 'Jolene', 'Anne',
                                      'Roxanne', 'Barbara', 'he', 'she', 'jeep', 'station_wagon', 'Mercedes', 'Camaro',
                                      'beach', 'home', 'airport', 'gate', 'restaurant', 'waiter', 'park'],
                                     np.arange(23)))

d_locations = dict(zip(['beach', 'airport', 'restaurant', 'bar', 'race', 'park'], np.arange(6)))

d_manners = dict(zip(['long', 'short', 'fast', 'free', 'pay', 'big', 'small', 'not', 'politely', 'obnoxiously'],
                     np.arange(10)))

d_attributes = dict(zip(['far', 'near', 'sunny', 'happy', 'raining', 'sad', 'cheap', 'expensive', 'clear', 'cloudy'],
                        np.arange(10)))


# Index to word dictionaries
i_agents = dict((v, k) for k, v in d_agents.iteritems())
i_predicates = dict((v, k) for k, v in d_predicates.iteritems())
i_patients_themes = dict((v, k) for k, v in d_patients_themes.iteritems())
i_recipients_destinations = dict((v, k) for k, v in d_recipients_destinations.iteritems())
i_locations = dict((v, k) for k, v in d_locations.iteritems())
i_manners = dict((v, k) for k, v in d_manners.iteritems())
i_attributes = dict((v, k) for k, v in d_attributes.iteritems())
