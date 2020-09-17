#!/usr/bin/env python

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import itertools

# Define the range of state lengths to iterate through. A different state length
# means a different number of classes.
STATE_LEN_RANGE = range(3,8)

# Make experiments repeatable.
RANDOM_STATE = 255
NEG_INF = -1000000

# Set how many students are in each grade.
STUDENT_INPUT = np.array([6, 19, 30, 28, 26, 31, 24], dtype=np.int64) # distance
#STUDENT_INPUT = np.array([12, 57, 61, 68, 63, 64, 58], dtype=np.int64) # hybrid

# Populate a list of students, represented by their grade level.
STUDENTS = []
for i in range(len(STUDENT_INPUT)):
    for j in range(STUDENT_INPUT[i]):
        STUDENTS.append(i)
STUDENTS = np.array(STUDENTS, dtype=np.int64)

# QUESTIONS
# What is the min number of kids per grade in a combo?

# Define a custom fitness function.
def klass_config_fitness(state, c):
    fitness = 0
    state = np.sort(state)
    klasses = np.array(np.split(STUDENTS, state))

    tk3_sum = 0
    tk3_count = 0
    for klass in klasses:
        klass_size = klass.shape[0]

        # Count values for grades TK-3.
        if 5 not in klass and 6 not in klass:
            tk3_sum += klass_size
            tk3_count += 1

        # Check that no class contains students for > 2 grades.
        if np.unique(klass).shape[0] > 2:
            return NEG_INF

        # Check that grades 4 and 5 don't exceed 34 students per class.
        if np.all(klass == 5) or np.all(klass == 6):
            if klass_size > 34:
                return NEG_INF

        # Check that 4/5 combos don't exceed 31 students per class.
        if np.array_equal(np.unique(klass), np.array([5,6])) and klass_size > 31:
            return NEG_INF

        # Check that 3/4 combos don't exceed 25 students per class.
        if np.array_equal(np.unique(klass), np.array([4,5])) and klass_size > 31:
            return NEG_INF

        # Check that 3/4 combos don't exceed 25 students per class.
        if np.array_equal(np.unique(klass), np.array([4,5])) and klass_size > 31:
            return NEG_INF

        fitness -= (31 - klass_size) ** 2

    # Check that the average class size for TK-3 doesn't exceed 25.
    if tk3_sum / tk3_count > 25:
        return NEG_INF

    return fitness

# Initialize the best state and that state's fitness.
# These are the ones to beat!
best_state_len = STATE_LEN_RANGE[0]
best_state = np.zeros(best_state_len, dtype=np.int64)
best_fitness = NEG_INF

# MAIN LOOP - Iterate through range of state lengths.
for state_len in STATE_LEN_RANGE:
    print('trying state_len', state_len, '...')

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

    # Run the Randomized Hill Climb algorithm.
    try_state, try_fitness = mlrose.random_hill_climb(
        problem,
        max_attempts=100,
        max_iters=1000,
        restarts=100,
        init_state=init_state,
        curve=False,
        random_state=RANDOM_STATE
    )
    print('try_fitness', try_fitness)

    # Set the best values.
    if try_fitness > best_fitness:
        best_fitness = try_fitness
        best_state = try_state
        best_state_len = state_len

print('')
print('best_state_len', best_state_len)
print('best_state', best_state)
print('best_fitness', best_fitness)
print('')

klasses = np.array(np.split(STUDENTS, np.sort(best_state)))
for klass in klasses:
    print(klass.shape[0], klass)
