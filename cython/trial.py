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


def trial(task, duration, trial_n=0, debugging=True,
          wholeFig=True):
    reset_history()
    reset_activities()

    prev_tr = 0
    moves = 0
    init_pos = task.trials[trial_n]["initial_pos_arm"]
    while True:
        for i in range(0, 500):
            iterate(dt)
            if i == 100:
                PPC.str.Iext.reshape((n_arm, n_targets))[init_pos, :] = 7
                M1_in.str.Iext.reshape((n_arm, n_sma))[init_pos, :] = 7

        trg = set_trial(task, trial=trial_n)

        for i in range(500, duration):
            iterate(dt)
            m1out = np.argmax(M1_out.str.U)
            if M1_out.str.U[m1out] > 30:
                sma = np.argmax(SMA.str.U)
                ppc = np.argmax(PPC.str.U)
                # ism = np.argmax(ISM.str.U)

                target_pos, final_pos = coord[trg],coord[m1out]
                task.records[trial_n]["move"] = m1out
                task.records[trial_n]["final_pos"] = final_pos
                task.records[trial_n]["target_pos"] = target_pos
                reward = rewards[trg, init_pos, m1out]
                task.records["reward"][trial_n] = reward
                task.process(RT=prev_tr + i - 500, debug=debugging)
                SMA_learning(reward, ppc, sma)
                # ISM_learning(trg, ism)

                task.records["SMAValues"][trial_n] = SMA_values
                task.records["PPCValues"][trial_n] = PPC_values
                task.records["Wppc_sma"][trial_n] = connections["PPC -> SMA"].weights
                task.records["Wsma_str"][trial_n] = connections["SMA -> STR"].weights
                task.records["Wppc_str"][trial_n] = connections["PPC -> STR"].weights
                # task.records["Wtrg_ism"][trial_n] = connections["TARGET -> ISM"].weights

                break
        if (m1out == trg) or (prev_tr +i + duration > duration_learning_positions):
            moves += 1
            task.records[trial_n]["moves"] = moves
            if debugging:
                task.debugging()
            return prev_tr +i
        else:
            prev_tr += i
            moves += 1
            init_pos = task.records[trial_n]["move"]
            reset_activities()


