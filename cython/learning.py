# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------

import numpy as np
import random
from trial import *
from parameters import *
import sys


def learning_trials(task, trials=n_learning_positions_trials, debugging=True, debug_simulation=False, duration=duration):
    if debug_simulation:
        steps = trials / 145
        print('  Starting   ', end=' ')

    for i in range(trials):

        trial(task, duration=duration, trial_n = i, debugging=debugging, wholeFig=True)

        if debug_simulation:
            if i % steps == 0:
                print('\b.', end=' ')
                sys.stdout.flush()

    if debug_simulation:
        print('   Done!')

    return
