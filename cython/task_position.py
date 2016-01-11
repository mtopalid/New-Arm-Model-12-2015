# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Nicolas P. Rougier, Meropi Topalidou
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import numpy as np


# -------------------------------------------------------------------- Task ---
class Task(object):
    """ A two-armed bandit task """

    def __init__(self, n=None, setup=True):
        self.trials = None
        self.records = None
        if setup:
            if n is None:
                self.setup()
            else:
                self.setup(n=n)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index < len(self.trials):
            return self.trials[self.index]
        raise StopIteration

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):

        if isinstance(index, slice):
            task = type(self)(setup=False)
            task.trials = self.trials[index]  # .copy()
            task.records = self.records[index]  # .copy()
            return task
        else:
            self.index = index
            return self.trials[index]

    def build(self, n=9):
        self.trials = np.zeros(n, [("initial_pos_arm", int, 1),
                                   ("target", int, 1)])
        self.records = np.zeros(n, [("best", float, 1),
                                    ("RTmove", float, 1),
                                    ("move", int, 1),
                                    ("moves", int, 1),
                                    ("reward", float, 1),
                                    ("target_pos", float, 2),
                                    ("final_pos", float, 2),
                                    ("SMAValues", float, 405),
                                    ("PPCValues", int, 405),
                                    ("Wppc_sma", float, 405),
                                    ("Wsma_str", float, 405),
                                    ("Wppc_str", float, 405)])#("Wtrg_ism", float, 81)

    def process(self, RT=0, debug=False):

        m = self.records[self.index]["final_pos"]
        t = self.records[self.index]["target_pos"]

        if (m == t).all():
            best = True
            # if debug:
            #     print("  Position has been reached")
        else:
            best = False
            # if debug:
            #     print("  Position has NOT been reached")

        # Record action, best action (was it the best action), reward and RT
        self.records[self.index]["best"] = best
        self.records[self.index]["RTmove"] = RT

    def debugging(self):
        print("Trial %d" % (self.index + 1))

        if self.records[self.index]["best"]:
            print("  Position has been reached")
        else:
            print("  Position has NOT been reached")

        P = self.records[:self.index + 1]["best"]
        print(("  Mean performance		: %.1f %%" % np.array(P * 100).mean()))
        r = self.records[self.index]["reward"]
        print(("  Reward    			: %f" % r))
        R = self.records[:self.index + 1]["reward"]
        print(("  Mean reward			: %.3f" % np.array(R).mean()))
        rt = self.records[self.index]["RTmove"]
        print(("  Response time	move    : %.3f ms" % np.array(rt)))
        initpos = self.trials[self.index]["initial_pos_arm"]
        print(("  Initial position      : %d" % initpos))
        trg = self.trials[self.index]["target"]
        print(("  Target                : %d" % trg))
        move = self.records[self.index]["move"]
        print(("  Chosen Move           : %d" % move))
        rt = self.records[:self.index + 1]["RTmove"]
        print(("  Mean Response time move: %.3f ms" % np.array(rt).mean()))
        moves = self.records[self.index]["moves"]
        print(("  Moves                 : %d" % moves))
        print("\n\n")

    def setup(self, n=81):

        # Make sure count is a multiple of 6
        n = (n // 81) * 81
        self.build(n)

        # n//9 x all combinations of initial positions of the arm
        # ipa = np.repeat(np.arange(9), n // 9)
        # np.random.shuffle(ipa)

        # Z = np.array([1, 3, 5, 7])
        #
        # # n//4 x all combinations of positions
        # temp = np.repeat(np.arange(4), n // 4)
        # np.random.shuffle(temp)
        # tar = Z[temp]
        # np.random.shuffle(tar[:])
        # tar = np.repeat(np.arange(9), n // 9)
        # np.random.shuffle(tar)

        cues = np.zeros((n, 2))
        temp = np.repeat(np.arange(9), n // 9)
        cues[:, 0] = 4#temp
        temp = np.arange(9)
        nn = n // 9
        for i in range(nn):
            cues[i * 9:(i + 1) * 9, 1] = temp
        np.random.shuffle(cues)

        for i in range(n):

            trial = self.trials[i]
            trial["initial_pos_arm"] = cues[i, 0]
            trial["target"] = cues[i, 1]


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    task = Task(n=81)

    for trial in task:
        # Only the associative group can provide (m1,c1), (m2,c2)
        # i1, i2 = (trial["ass"].ravel().argsort())[-2:]
        # m1, c1 = np.unravel_index(i1, (4, 4))
        # m2, c2 = np.unravel_index(i2, (4, 4))

        # Reward probabilities
        # r1, r2 = trial["rwd"][c1], trial["rwd"][c2]

        # Random action
        # if random.uniform(0,1) < 0.5: action = m1
        # else:                         action = m2

        # Best action
        # if r1 > r2:
        #     action = m1
        # else:
        #     action = m2
        #
        # reward, best = task.process(trial, action=action, debug=True)
        print(task.trials["initial_pos_arm"])
        print()
        print(task.trials["target"])
        print()
