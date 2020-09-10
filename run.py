#!/usr/bin/env python

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np

#                    t   k   1   2   3   4   5
STUDENTS = np.array([6,  19, 30, 28, 26, 31, 24], dtype=np.int64) # distance
#STUDENTS = np.array([12, 57, 61, 68, 63, 64, 58], dtype=np.int64) # hybrid

GRADES = 'tk12345'
STATE_LEN = (STUDENTS.shape[0] * 2 - 1) * 3
INIT_STATE = np.linspace(0, np.sum(STUDENTS), STATE_LEN).astype(np.int64)
print(INIT_STATE)
exit()

# Define a custom fitness function.
def class_config_fitness(state, c):
    student_str = ''
    for i in range(len(STUDENTS)):
        student_str += GRADES[i] * STUDENTS[i]
    return len(student_str) * c

# Initialize the fitness function object.
kwargs = {'c': 10}
fitness = mlrose.CustomFitness(class_config_fitness, problem_type='discrete', **kwargs)
init_state = np.zeros(STATE_LEN, dtype=np.int64)
print(STATE_LEN)
print(fitness.evaluate(init_state))
exit()

# Initialize the discrete problem object.
problem = mlrose.DiscreteOpt(
    length=init_state.shape[0],
    fitness_fn=fitness,
    maximize=False,
    max_val=8
)

#
#best_state, best_fitness = random_hill_climb(
#    problem,
#    max_attempts=10,
#    max_iters=10,
#    restarts=0,
#    init_state=None,
#    curve=False,
#    random_state=None
#)
