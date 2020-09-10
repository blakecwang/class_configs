#!/usr/bin/env python

import numpy as np

# 0 -> 0,1
# 2 -> 3,4,5
# 6 -> 11,12

GRADE_NAMES = ['tk', 'tk-k', 'k', 'k-1st', '1st', '1-2nd', '2nd', '2nd-3rd', '3rd', '3rd-4th', '4th', '4th-5th', '5th']

# Configuration MUST
# - TK-3 must have average class size of 25
# - 4,5 hard cap at 34
# - 3/4 combo cap at 25
# - 4/5 combo cap at 31
#
# Configuration SHOULD
# - maximize utilization
# - minimize number of combos

# Can I handle distance and hybrid totally separate? YES!

#                   tk  k   1   2   3   4   5
distance_students = [6,  19, 30, 28, 26, 31, 24]
hybrid_students   = [12, 57, 61, 68, 63, 64, 58]

def run(students):
    state = np.zeros((len(GRADE_NAMES),3), dtype=np.int8)
    student_grades = len(students)

run(distance_students)
