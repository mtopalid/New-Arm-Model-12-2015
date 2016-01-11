# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------

# Evolution of single trial with Guthrie protocol
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Include to the path files from cython folder
    t = '../cython/'
    import sys

    sys.path.append(t)
    # model file build the structures and initialize the model
    from model import *
    from trial import *
    from task_position import Task
    import os
    from learning import *

    folder = '../Results/Learn_Positions'  # '#+M1_learning
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(100):
        print("\nSimulation: %d\n" % (i + 1))

        # Initialize the system
        task = Task(n=n_learning_positions_trials)
        #
        # Save the trials
        f = folder + '/Task%03d.npy' % (i + 1)
        np.save(f, task.trials)

        # Repeated trials with learning after each trial
        learning_trials(task, trials=n_learning_positions_trials, duration=duration, debug_simulation=True,
                        debugging=False)  # , tr=tr-1)

        np.set_printoptions(threshold=3)
        P = task.records["best"]
        print("  Mean performance		: %.1f %%" % np.array(P * 100).mean())
        R = task.records["reward"]
        print("  Mean reward			: %.3f" % np.array(R).mean())
        # print "Moves:\n", task.records["moves"][:n_learning_positions_trials]

        f = folder + '/Records%03d.npy' % (i + 1) + ''
        np.save(f, task.records)
