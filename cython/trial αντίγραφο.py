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
            M1_in.str.Iext.reshape((n_arm,n_sma))[init_pos,:] \
                = 7

    set_trial(task, trial=trial_n)

    for i in range(500, duration):
        iterate(dt)
    return i

def trial1(task, cues_pres=True, ncues=2, duration=duration, learn=True, debugging=True, debugging_arm=True, trial_n=0,
          wholeFig=False):
    reset_activities()
    reset_history()
    # ct = None
    # cog_time = None
    mot_time = None
    choice_made = False
    target = None
    t1 = 0
    t2 = 0
    pos = [4, 4]
    moves = 0
    for i in range(0, 500):
        iterate(dt)
        # if CTX.cog.delta > 20 and not ct:
        #     ct = 1
        # if CTX.cog.delta > decision_threshold and not cog_time:
        #     cog_time = i - 500

        # Put the arm in a position after the stabilization of the structures
        if i == 200:
            # pos = np.random.randint(n_arm, size=2)
            task.trials["initial_pos"][trial_n][:] = pos[:]

            # pos = task.trials["initial_pos"][trial_n].copy()
            ARM.theta1.Iext[pos[0]] = 17
            ARM.theta2.Iext[pos[1]] = 17
            # ARM.theta1.Iext[4] = 17
            # ARM.theta2.Iext[4] = 17

    if cues_pres:
        set_trial(task, num=ncues, trial=trial_n)

    for i in range(500, duration):
        iterate(dt)

        # Check if a target position was chosen
        if not choice_made:
            if CTX.mot.delta > decision_threshold and not mot_time:
                mot_time = i - 500
                task.records["RTmot"][trial_n] = mot_time
                mot_choice = np.argmax(CTX.mot.U)
                target = buttons[mot_choice, :]
                choice_made = True

        arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]

        # Check if the first angle of the arm was changed
        if ((arm[0] != pos[0] and ARM.theta1.delta > 0.5) or (
                        np.argmax(
                            SMA.theta1.V) == 8 and SMA.theta1.delta > 5.)) and t1 == 0 and choice_made:  # or i == t1 + 3000):
            pos[0] = arm[0]
            t1 = 1

        # Check if the second angle of the arm was changed
        if ((arm[1] != pos[1] and ARM.theta2.delta > 0.5) or (
                        np.argmax(
                            SMA.theta2.V) == 8 and SMA.theta2.delta > 5.)) and t2 == 0 and choice_made:  # or i == t2 + 3000):  #
            pos[1] = arm[1]
            t2 = 1

        if t1 == 1 and t2 == 1:

            moves += 1
            task.records["final_pos"][trial_n] = pos
            task.records["target_pos"][trial_n] = target
            t = i  # - 500

            # Learning
            # np.argmax(...): which neuron is more active in the structure
            reward = rewards[task.trials["initial_pos"][trial_n][0], task.trials["initial_pos"][trial_n][1], pos[0], pos[1]]
            SMA_learning1(reward, np.argmax(PPC.theta1.V), np.argmax(SMA.theta1.V))
            SMA_learning2(reward, np.argmax(PPC.theta2.V), np.argmax(SMA.theta2.V))
            # M1_learning1(np.argmax(M1.theta1.V), np.argmax(M1.theta2.V))
            # M1_learning2(np.argmax(M1.theta1.V), np.argmax(M1.theta2.V))

            if (arm == target).all():

                # Save the results of the trial
                for j in range(n):
                    if (np.array(arm) == buttons[j, :]).all():
                        move = j
                        break
                task.records["move"][trial_n] = move
                task.records["best"][trial_n] = True
                task.records["reward"][trial_n] = reward
            else:
                task.records["move"][trial_n] = 4
                task.records["best"][trial_n] = False

            task.records["reward"][trial_n] = reward
            task.records["moves"][trial_n] = moves

            if debugging_arm:
                print("Reward: ", reward)
                debug_arm()

            task.process(task[trial_n], debug=debugging, RT=t - 500)

            task.records["SMAValues1"][trial_n] = SMA_value_th1
            task.records["PPCValues1"][trial_n] = PPC_value_th1
            task.records["SMAValues2"][trial_n] = SMA_value_th2
            task.records["PPCValues2"][trial_n] = PPC_value_th2
            task.records["Wppc_sma1"][trial_n] = connections["PPC.theta1 -> SMA.theta1"].weights
            task.records["Wsma_str1"][trial_n] = connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights
            task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights
            task.records["Wppc_sma2"][trial_n] = connections["PPC.theta2 -> SMA.theta2"].weights
            task.records["Wsma_str2"][trial_n] = connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights
            task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights
            # task.records["Wm1_1"][trial_n] = connections["M1.theta1 -> M1.theta2"].weights
            # task.records["Wm1_2"][trial_n] = connections["M1.theta2 -> M1.theta1"].weights
            return t

    # Save results if a move hasn't occurred
    t = duration
    task.records["best"][trial_n] = False
    task.records["move"][trial_n] = 4
    task.records["SMAValues1"][trial_n] = SMA_value_th1
    task.records["PPCValues1"][trial_n] = PPC_value_th1
    task.records["SMAValues2"][trial_n] = SMA_value_th2
    task.records["PPCValues2"][trial_n] = PPC_value_th2
    task.records["Wppc_sma1"][trial_n] = connections["PPC.theta1 -> SMA.theta1"].weights
    task.records["Wsma_str1"][trial_n] = connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights
    task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights
    task.records["Wppc_sma2"][trial_n] = connections["PPC.theta2 -> SMA.theta2"].weights
    task.records["Wsma_str2"][trial_n] = connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights
    task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights
    # task.records["Wm1_1"][trial_n] = connections["M1.theta1 -> M1.theta2"].weights
    # task.records["Wm1_2"][trial_n] = connections["M1.theta2 -> M1.theta1"].weights
    task.records["moves"][trial_n] = moves

    return t


def trial_continuous(task, cues_pres=True, ncues=2, duration=duration, learn=True,                       debugging=True, debugging_arm=True,
                      trial_n=0, wholeFig=False):
    reset_activities()
    reset_history()
    # ct = None
    # cog_time = None
    mot_time = None
    choice_made = False
    target = None
    reset_arm_pos = 0
    t1 = 0
    t2 = 0
    pos = np.array([4, 4])
    moves = 0
    positions = []

    for i in range(0, 500):
        iterate(dt)
        # if CTX.cog.delta > 20 and not ct:
        #     ct = 1
        # if CTX.cog.delta > decision_threshold and not cog_time:
        #     cog_time = i - 500

        # Put the arm in a position after the stabilization of the structures

        if i == 200:
            # pos = np.random.randint(n_arm, size=2)
            task.trials["initial_pos"][trial_n] = pos
            temp = pos.copy()
            positions.append(temp)

            # pos = task.trials["initial_pos"][trial_n].copy()
            ARM.theta1.Iext[pos[0]] = 17
            ARM.theta2.Iext[pos[1]] = 17
            # ARM.theta1.Iext[4] = 17
            # ARM.theta2.Iext[4] = 17

    if cues_pres:
        m = set_trial(task, num=ncues, trial=trial_n)

    for i in range(500, duration):
        iterate(dt)

        if not choice_made:
            # Test if a decision has been made
            if CTX.mot.delta > decision_threshold and not mot_time:
                mot_time = i - 500
                task.records["RTmot"][trial_n] = mot_time
                mot_choice = np.argmax(CTX.mot.U)
                target = buttons[mot_choice, :]
                choice_made = True

        arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]

        # Check if the first angle of the arm was changed
        if ((arm[0] != pos[0] and ARM.theta1.delta > 0.5) or (
                        np.argmax(
                            SMA.theta1.V) == 8 and SMA.theta1.delta > 5.)) and t1 == 0 and choice_made:  # or i == t1 + 3000):
            t1 = 1

        # Check if the second angle of the arm was changed
        if ((arm[1] != pos[1] and ARM.theta2.delta > 0.5) or (
                        np.argmax(
                            SMA.theta2.V) == 8 and SMA.theta2.delta > 5.)) and t2 == 0 and choice_made:  # or i == t2 + 3000):  #
            t2 = 1

        if t1 == 1 and t2 == 1:

            moves += 1

            reward = rewards[pos[0], pos[1], arm[0], arm[1]]
            pos[0] = arm[0]
            pos[1] = arm[1]
            temp = pos.copy()
            positions.append(temp)
            # Learning
            # np.argmax(...): which neuron is more active in the structure
            SMA_learning1(reward, np.argmax(PPC.theta1.V), np.argmax(SMA.theta1.V))
            SMA_learning2(reward, np.argmax(PPC.theta2.V), np.argmax(SMA.theta2.V))
            # M1_learning1(np.argmax(M1.theta1.V), np.argmax(M1.theta2.V))
            # M1_learning2(np.argmax(M1.theta1.V), np.argmax(M1.theta2.V))

            if debugging_arm:  # 0:
                print("Reward: ", reward)
                debug_arm()

            # Check if the target has been reached
            if (arm == target).all():

                # Save the results of the trial
                for j in range(n):
                    if (np.array(arm) == buttons[j, :]).all():
                        move = j
                        break
                task.records["move"][trial_n] = move
                task.records["best"][trial_n] = True
                task.records["reward"][trial_n] = reward
                task.records["moves"][trial_n] = moves
                task.records["final_pos"][trial_n] = pos
                task.records["target_pos"][trial_n] = target

                t = i  # - 500
                task.process(task[trial_n], debug=debugging, RT=t - 500)
                print("Moves: ", end=' ')
                for i in range(len(positions)):
                    if i == len(positions)-1:
                        print(positions[i])
                    else:
                        print(positions[i], " -> ", end=' ')
                print()

                task.records["SMAValues1"][trial_n] = SMA_value_th1
                task.records["PPCValues1"][trial_n] = PPC_value_th1
                task.records["SMAValues2"][trial_n] = SMA_value_th2
                task.records["PPCValues2"][trial_n] = PPC_value_th2
                task.records["Wppc_sma1"][trial_n] = connections["PPC.theta1 -> SMA.theta1"].weights
                task.records["Wsma_str1"][trial_n] = connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights
                task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights
                task.records["Wppc_sma2"][trial_n] = connections["PPC.theta2 -> SMA.theta2"].weights
                task.records["Wsma_str2"][trial_n] = connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights
                task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights
                # task.records["Wm1_1"][trial_n] = connections["M1.theta1 -> M1.theta2"].weights
                # task.records["Wm1_2"][trial_n] = connections["M1.theta2 -> M1.theta1"].weights
                return t
            else:
                reset_arm1_activities()
                t1 = 0
                reset_arm2_activities()
                t2 = 0
                reset_arm_pos = i

        if i == reset_arm_pos + 10:
            ARM.theta1.Iext[pos[0]] = 17
            ARM.theta2.Iext[pos[1]] = 17


    # Save if the target hasn't been reached
    task.records["best"][trial_n] = False
    task.records["move"][trial_n] = 4
    task.records["moves"][trial_n] = moves + 1
    task.records["final_pos"][trial_n] = pos
    task.records["target_pos"][trial_n] = target

    t = i  # - 500
    task.process(task[trial_n], debug=debugging, RT=t - 500)
    print("Moves: ", end=' ')
    for i in range(len(positions)):
        if i == len(positions)-1:
            print(positions[i])
        else:
            print(positions[i], " -> ", end=' ')
    print()
    task.records["SMAValues1"][trial_n] = SMA_value_th1
    task.records["PPCValues1"][trial_n] = PPC_value_th1
    task.records["SMAValues2"][trial_n] = SMA_value_th2
    task.records["PPCValues2"][trial_n] = PPC_value_th2
    task.records["Wppc_sma1"][trial_n] = connections["PPC.theta1 -> SMA.theta1"].weights
    task.records["Wsma_str1"][trial_n] = connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights
    task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights
    task.records["Wppc_sma2"][trial_n] = connections["PPC.theta2 -> SMA.theta2"].weights
    task.records["Wsma_str2"][trial_n] = connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights
    task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights
    # task.records["Wm1_1"][trial_n] = connections["M1.theta1 -> M1.theta2"].weights
    # task.records["Wm1_2"][trial_n] = connections["M1.theta2 -> M1.theta1"].weights
    task.records["moves"][trial_n] = moves

    return t
