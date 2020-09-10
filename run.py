#!/usr/bin/env python

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np

# Define a custom fitness function.
def class_config_fitness(state, c):
    return c*np.sum(state)

# Set the initial state.
# States are lists of classes sizes.
init_state = np.array([
    # tk_hybrid
    0,0,0,
    # tk_distance
    0,0,0,
    # tk_k_hybrid
    0,0,0,
    # tk_k_distance
    0,0,0,
    # k_hybrid
    0,0,0,
    # k_distance
    0,0,0,
    # k_1_hybrid
    0,0,0,
    # k_1_distance
    0,0,0,
    # 1_hybrid
    0,0,0,
    # 1_distance
    0,0,0,
    # 1_2_hybrid
    0,0,0,
    # 1_2_distance
    0,0,0,
    # 2_hybrid
    0,0,0,
    # 2_distance
    0,0,0,
    # 2_3_hybrid
    0,0,0,
    # 2_3_distance
    0,0,0,
    # 3_hybrid
    0,0,0,
    # 3_distance
    0,0,0,
    # 3_4_hybrid
    0,0,0,
    # 3_4_distance
    0,0,0,
    # 4_hybrid
    0,0,0,
    # 4_distance
    0,0,0,
    # 4_5_hybrid
    0,0,0,
    # 4_5_distance
    0,0,0,
    # 5_hybrid
    0,0,0,
    # 5_distance
    0,0,0
])

#                tk  k   1   2   3   4   5
dist_students = [6,  19, 30, 28, 26, 31, 24]
hybd_students = [12, 57, 61, 68, 63, 64, 58]

# Initialize the fitness function object.
kwargs = {'c': 10}
fitness = mlrose.CustomFitness(class_config_fitness, problem_type='discrete', **kwargs)
print(init_state.shape[0])
#print(fitness.evaluate(init_state))

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
