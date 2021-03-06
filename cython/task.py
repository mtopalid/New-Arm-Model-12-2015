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

    def build(self, n, nsma=17, nppc=36, narm=9):
        self.trials = np.zeros(n, [("initial_pos", float, 2),
                                   ("mot", float, 4),
                                   ("cog", float, 4),
                                   ("ass", float, (4, 4)),
                                   ("rwd", float, 4)])
        self.records = np.zeros(n, [("action", float, 1),
                                    ("move", float, 1),
                                    ("shape", float, 1),
                                    ("cog_choice", float, 1),
                                    ("best", float, 1),
                                    ("RTcog", float, 1),
                                    ("RTmot", float, 1),
                                    ("RTmove", float, 1),
                                    ("CueValues", float, 4),
                                    ("Wstr", float, 4),
                                    ("Wcog", float, 4),
                                    ("Wmot", float, 4),
                                    ("SMAValues1", float, nsma * nppc),
                                    ("PPCValues1", float, nsma * nppc),
                                    ("SMAValues2", float, nsma * nppc),
                                    ("PPCValues2", float, nsma * nppc),
                                    ("Wppc_sma1", float, nsma * nppc),
                                    ("Wsma_str1", float, nsma * nppc),
                                    ("Wppc_str1", float, nsma * nppc),
                                    ("Wppc_sma2", float, nsma * nppc),
                                    ("Wsma_str2", float, nsma * nppc),
                                    ("Wppc_str2", float, nsma * nppc),
                                    ("Wm1_1", float, (nsma * narm)**2),
                                    ("Wm1_2", float, (nsma * narm)**2),
                                    ("moves", int, 1),
                                    ("reward", float, 1),
                                    ("target_pos", float, 2),
                                    ("final_pos", float, 2)])

    def process(self, trial, action, RT=0, debug=False):

        # Only the associative feature can provide (m1,c1) and (m2,c2)
        i1, i2 = (trial["ass"].ravel().argsort())[-2:]
        m1, c1 = np.unravel_index(i1, (4, 4))
        m2, c2 = np.unravel_index(i2, (4, 4))
        if c2 < c1:
            c1, c2 = c2, c1
            m1, m2 = m2, m1
        r1, r2 = trial["rwd"][c1], trial["rwd"][c2]
        m = self.records[self.index]["move"]

        if debug:
            print(("Trial %d" % (self.index + 1)))
            print(("  Action                : %d " % action))
        if m == m1:
            reward = np.random.uniform(0, 1) < trial["rwd"][c1]
            self.records[self.index]["shape"] = c1
            best = r1 > r2
            if debug:
                print(("  Move			        : [%d] / %d" % (m1, m2)))
                if best:
                    print(("  Choice			    : [%d] / %d  (good)" % (c1, c2)))
                else:
                    print(("  Choice			    : [%d] / %d  (bad)" % (c1, c2)))
                print(("  Reward (p=%.2f)		: %d" % (trial["rwd"][c1], reward)))
        elif m == m2:
            reward = np.random.uniform(0, 1) < trial["rwd"][c2]
            self.records[self.index]["shape"] = c2
            best = r2 > r1
            if debug:
                print(("  Move			        : %d / [%d]" % (m1, m2)))
                if best:
                    print(("  Choice			    : %d / [%d] (good)" % (c1, c2)))
                else:
                    print(("  Choice			    : %d / [%d] (bad)" % (c1, c2)))
                print(("  Reward (p=%.2f)		: %d" % (trial["rwd"][c2], reward)))
        else:
            reward = 0.0
            best = False
            if debug:
                print(("  Choice			    : %d /  %d  (bad)" % (c1, c2)))
                print("  No Reward")

        # Record action, best action (was it the best action), reward and RT
        self.records[self.index]["action"] = action
        self.records[self.index]["best"] = best
        self.records[self.index]["RTmove"] = RT
        self.records[self.index]["reward"] = reward

        if debug:
            P = self.records[:self.index + 1]["best"]
            print(("  Mean performance		: %.1f %%" % np.array(P * 100).mean()))
            R = self.records[:self.index + 1]["reward"]
            print(("  Mean reward			: %.3f" % np.array(R).mean()))
            rt = self.records[self.index]["RTmove"] - self.records[self.index]["RTmot"]
            print(("  Response time	move    : %.3f ms" % np.array(rt)))
            n_moves = self.records[self.index]["moves"]
            print(("  Number of moves       : %d" % n_moves))
            rt = self.records[:self.index + 1]["RTmove"]
            print(("  Mean Response time	move: %.3f ms" % np.array(rt).mean()))
            rt = self.records[:self.index + 1]["RTmot"]
            print(("  Mean Response time	mot: %.3f ms" % np.array(rt).mean()))
            rt = self.records[:self.index + 1]["RTcog"]
            print(("  Mean Response time	cog: %.3f ms" % np.array(rt).mean()))

        return reward, best


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    from task_a import Task_A
    # from task_b import Task_B
    # from task_cg import Task_C
    # from task_cp import Task_C
    from task_1ch import Task_1ch

    # task = Task_A(n=180)
    # task = Task_B(n=120)
    # task = Task_C(n=30, r=6)
    task = Task_1ch(n=81)

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
        task.trials["initial_pos"]
