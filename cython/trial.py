# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#				Nicolas Rougier  (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
from model import *
from kinematics import *
from parameters import *

def trial(task,duration, trial_n=0, debugging=True,
                 wholeFig=True):
    reset_activities()
    reset_history()
    for i in range(0, 500):
        iterate(dt)
        if i == 100:
            init_pos = task.trials[trial_n]["initial_pos_arm"]
            PPC.str.Iext.reshape((n_arm,n_targets))[init_pos,:] = 7
            M1_in.str.Iext.reshape((n_arm, n_sma))[init_pos,:] = 7

    set_trial(task, trial=trial_n)

    for i in range(500, duration):
        iterate(dt)
    return i
