#!/usr/bin/env python

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import itertools

# Make experiments repeatable.
RANDOM_STATE = 255

# Set how many students are in each grade.
STUDENT_INPUT = np.array([6, 19, 30, 28, 26, 31, 24], dtype=np.int64) # distance
#STUDENT_INPUT = np.array([12, 57, 61, 68, 63, 64, 58], dtype=np.int64) # hybrid

# Populate a list of students, represented by their grade level.
STUDENTS = []
for i in range(len(STUDENT_INPUT)):
    for j in range(STUDENT_INPUT[i]):
        STUDENTS.append(i)
STUDENTS = np.array(STUDENTS, dtype=np.int64)

# Define the range of state lengths to iterate through. A different state length
# means a different number of classes.
STATE_LEN_RANGE = range(3,5)

# Define a custom fitness function.
def klass_config_fitness(state, c):
    fitness = 0
    state = np.sort(state)
    klasses = np.array(np.split(STUDENTS, state))

    for klass in klasses:
        klass_size = klass.shape[0]
        fitness -= (31 - klass_size) ** 2

    return fitness

# Iterate through range of state lengths.
for state_len in STATE_LEN_RANGE:
    # Build the initial state by dividing the STUDENTS array evenly.
    # Each state is an array of indices for splitting the STUDENTS array.
    init_state = np.linspace(0, np.sum(STUDENT_INPUT) - 1, state_len).astype(np.int64)

    # Initialize the fitness function object.
    kwargs = {'c': 10}
    fitness = mlrose.CustomFitness(klass_config_fitness, problem_type='discrete', **kwargs)

    # Initialize the discrete problem object.
    problem = mlrose.DiscreteOpt(
        length=init_state.shape[0],
        fitness_fn=fitness,
        maximize=True,
        max_val=STUDENTS.shape[0]
    )

    best_state, best_fitness = mlrose.random_hill_climb(
        problem,
        max_attempts=100,
        max_iters=1000,
        restarts=100,
        init_state=init_state,
        curve=False,
        random_state=RANDOM_STATE
    )

    print('')
    print('state_len', state_len)
    print('--')
    print('init_state', init_state)
    print('init_state fitness', fitness.evaluate(init_state))
    print('--')
    print('best_state', best_state)
    print('best_state fitness', fitness.evaluate(best_state))

    #klasses = np.array(np.split(STUDENTS, np.sort(best_state)))
    #for klass in klasses:
    #    print(klass.shape[0])
