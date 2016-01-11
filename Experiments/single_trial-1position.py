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

def single_trial():
    # 1 if there is presentation of cues else 0
    # cues_pres = 1
    # trials = 1
    # Define the shapes and the positions that we'll be used to each trial
    # n should be multiple of 6 because there are 6 valuable combinations of shapes and positions
    task = Task()

    # Compute a single trial
    time = trial(task, duration=duration, debugging=True,
                 wholeFig=True)
    # print "Moves        : ", task.records[0]["moves"]
    # print "Time         : ", time

    # retrieve the activity history of the structures
    histor = history(dur=time)
    ppc = histor["PPC"]["str"]#[:time]
    sma = histor["SMA"]["str"]#[:time]
    trg = histor["TARGET"]["str"]#[:time]
    ism = histor["ISM"]["str"]#[:time]
    m1in = histor["M1_in"]["str"]#[:time]
    m1out = histor["M1_out"]["str"]#[:time]
    str = histor["STR"]["str"]#[:time]
    # stn = histor["STN"]["str"][:time]
    thl = histor["THL"]["str"]#[:time]
    # gpi = histor["GPI"]["str"][:time]


    plt.figure(figsize=(18, 9))
    plt.subplot(331)
    plt.plot(ppc)
    plt.ylim(ppc.min() - 10, ppc.max() + 10)
    plt.xlim(-10, +5010)
    plt.title('PPC')

    plt.subplot(334)
    plt.plot(sma[:, 0], c='b', label="Up")
    plt.plot(sma[:, 1], c='r', label="Down")
    plt.plot(sma[:, 2], c='g', label="Stay")
    plt.plot(sma[:, 3], c='m', label="Right")
    plt.plot(sma[:, 4], c='c', label="Left")
    plt.legend()
    plt.ylim(sma.min() - 10, sma.max() + 10)
    # plt.xlim(-100, +5100)
    plt.title('SMA')

    plt.subplot(336)
    plt.plot(trg[:, 0], c='b', label=1)
    plt.plot(trg[:, 1], c='r', label=2)
    plt.plot(trg[:, 2], c='g', label=3)
    plt.plot(trg[:, 3], c='m', label=4)
    plt.plot(trg[:, 4], c='c', label=5)
    plt.plot(trg[:, 5], c='y', label=6)
    plt.plot(trg[:, 6], c='k', label=7)
    plt.plot(trg[:, 7], 'bo', label=8)
    plt.plot(trg[:, 8], 'r*', label=9)
    # plt.legend()
    plt.title('Target')
    #
    #
    plt.subplot(332)
    plt.plot(m1in)
    plt.title('M1in')

    plt.subplot(335)
    plt.plot(m1out[:, 0], c='b', label=0)
    plt.plot(m1out[:, 1], c='r', label=1)
    plt.plot(m1out[:, 2], c='g', label=2)
    plt.plot(m1out[:, 3], c='m', label=3)
    plt.plot(m1out[:, 4], c='c', label=4)
    plt.plot(m1out[:, 5], c='y', label=5)
    plt.plot(m1out[:, 6], c='k', label=6)
    plt.plot(m1out[:, 7], 'bo', label=7)
    plt.plot(m1out[:, 8], 'r*', label=8)
    plt.title('M1out')
    #
    #
    #
    plt.subplot(333)
    plt.plot(ism[:, 0], c='b', label=1)
    plt.plot(ism[:, 1], c='r', label=2)
    plt.plot(ism[:, 2], c='g', label=3)
    plt.plot(ism[:, 3], c='m', label=4)
    plt.plot(ism[:, 4], c='c', label=5)
    plt.plot(ism[:, 5], c='y', label=6)
    plt.plot(ism[:, 6], c='k', label=7)
    plt.plot(ism[:, 7], 'bo', label=8)
    plt.plot(ism[:, 8], 'r*', label=9)
    plt.title('ISM')

    plt.subplot(337)
    init_pos = task.trials[0]["initial_pos_arm"]
    prop = np.ones(trg.shape[0]) * init_pos
    plt.plot(prop)
    plt.title('Propioceptors')

    # plt.subplot(337)
    # plt.plot(stn)
    # plt.title('stn')
    # plt.ylim(stn.min()-10, stn.max()+10)
    # plt.xlim(-100, +5100)

    plt.subplot(338)
    plt.plot(str)
    plt.title('str')
    plt.ylim(str.min() - 10, str.max() + 10)
    # plt.xlim(-100, +5100)

    plt.subplot(339)
    plt.plot(thl)
    plt.title('Thalamus')
    plt.ylim(thl.min() - 10, thl.max() + 10)
    # plt.xlim(-100, +5100)

    # plt.subplot(335)
    # plt.plot(gpi)
    # plt.title('GPi')
    # plt.ylim(gpi.min()-100, gpi.max()+100)
    # plt.xlim(-100, +5100)
    plt.show()


if __name__ == "__main__":
    # Include to the path files from cython folder
    temp = '../cython/'
    import sys

    sys.path.append(temp)
    # model file build the structures and initialize the model
    from model import *
    from display import *
    from trial import *
    from task_position import Task

    single_trial()
