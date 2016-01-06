# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------

from c_dana import *
from parameters import *
from kinematics import *

clamp = Clamp(min=0, max=1000)
sigmoid = Sigmoid(Vmin=Vmin, Vmax=Vmax, Vh=Vh, Vc=Vc)

# Build structures
TARGET = Structure(tau=tau, rest=TARGET_rest, noise=Cortex_N, activation=clamp, n=n_targets)
PPC = Structure(tau=tau, rest=PPC_rest, noise=Cortex_N, activation=clamp, n=n_ppc)
SMA = Structure(tau=tau, rest=SMA_rest, noise=Cortex_N, activation=clamp, n=n_sma)
# BG
STR = Structure(tau=tau, rest=STR_rest, noise=Striatum_N, activation=clamp, n=n_sma * n_ppc)
STN = Structure(tau=tau, rest=STN_rest, noise=STN_N, activation=clamp, n=n_sma * n_ppc)
GPE = Structure(tau=tau, rest=GPE_rest, noise=GPe_N, activation=clamp, n=n_sma)
GPI = Structure(tau=tau, rest=GPI_rest, noise=GPi_N, activation=clamp, n=n_sma)
THL = Structure(tau=tau, rest=THL_rest, noise=Thalamus_N, activation=clamp, n=n_sma)


structures = (PPC, SMA, TARGET, STR, STN, GPE, GPI, THL)  #


# Add noise to weights
def weights(shape, s=0.005, initial=0.5):
    N = np.random.normal(initial, s, shape)
    N = np.minimum(np.maximum(N, 0.0), 1.0)
    return Wmin + (Wmax - Wmin) * N


def Wlateral(n):
    return (2 * np.eye(n) - np.ones((n, n))).ravel()


def Wppc2sma(n1=n_sma, n2=n_arm, n3=n_targets):
    w = weights(n1 * n2 * n3).reshape((n1, n2, n3))
    # w = np.ones((n1, n2, n3))
    w[0, :3, :] = 0
    w[1, -3:, :] = 0
    w[3, 2::3, :] = 0
    w[4, ::3, :] = 0

    return w.reshape(n1 * n2 * n3)

def WM1in2M1out(n1=n_arm, n2=n_sma, n3=n_m1_out):
    w = np.zeros((n3, n1, n2))
    for i in range(n3):
        w[i][np.where(Wm1in2mout==i)] = 1
    return w.reshape(n3*n1*n2)
# np.set_printoptions(threshold='nan')
# print(WM1in2M1out().reshape((n_m1_out, n_arm, n_sma)))
# print(Wppc2sma().reshape((n_sma, n_arm, n_targets)))
# Connectivity 
connections = {

    # Lateral connectivity

    "PPC -> PPC": AllToAll(PPC.str.V, PPC.str.Isyn, Wlateral(n_ppc)),

    "SMA -> SMA": AllToAll(SMA.str.V, SMA.str.Isyn, Wlateral(n_sma)),

    # "STR -> STR": AllToAll(STR.str.V, STR.str.Isyn, Wlateral(n_sma * n_ppc)),


    # Input To SMA
    "PPC -> SMA": AllToAll(PPC.str.V, SMA.str.Isyn, Wppc2sma(), np.array([n_sma, n_arm, n_targets])),#weights(n_sma*n_ppc)

    # Input to PPC
    "TARGET -> PPC": OneToColumn(TARGET.str.V, PPC.str.Isyn, np.ones(n_targets*n_arm), np.array([n_arm, n_targets])),

    # SMA <-> BG
    # "SMA.str -> Str.str": OneToOne(SMA.str.V, Str.str.Isyn, np.ones(n_sma)),
    # "Str.str -> GPE.str": OneToOne(Str.str.V, GPE.str.Isyn, np.ones(n_sma)),
    # "Str.str -> GPI.str": OneToOne(Str.str.V, GPI.str.Isyn, np.ones(n_sma * n_ppc)),


    "SMA.str -> STN.str": OneToRow(SMA.str.V, STN.str.Isyn, np.ones(n_sma* n_ppc), np.array([n_sma, n_ppc])),
    "SMA.str -> STR.str": OneToRow(SMA.str.V, STR.str.Isyn, weights(n_sma * n_ppc), np.array([n_sma, n_ppc])),
    # plastic (RL)
    "PPC.str -> STR.str": OneToColumn(PPC.str.V, STR.str.Isyn, 0.5 * np.ones(n_sma * n_ppc), np.array([n_sma, n_ppc])),
    "PPC.str -> STN.str": OneToColumn(PPC.str.V, STN.str.Isyn, np.ones(n_sma* n_ppc), np.array([n_sma, n_ppc])),
    # plastic (RL)

    "STR.str -> GPE.str": RowToOne(STR.str.V, GPE.str.Isyn, np.ones(n_sma * n_ppc), np.array([n_sma, n_ppc])),
    "STR.str -> GPI.str": RowToOne(STR.str.V, GPI.str.Isyn, np.ones(n_sma * n_ppc), np.array([n_sma, n_ppc])),
    # #
    "GPE.str -> STN.str": OneToRow(GPE.str.V, STN.str.Isyn, np.ones(n_sma * n_ppc), np.array([n_sma, n_ppc])),
    "STN.str -> GPI.str": AllToAll(STN.str.V, GPI.str.Isyn, np.ones(n_sma * n_sma * n_ppc)),

    "GPI.str -> THL.str": OneToOne(GPI.str.V, THL.str.Isyn, np.ones(n_sma)),
    "THL.str -> SMA.str": OneToOne(THL.str.V, SMA.str.Isyn, np.ones(n_sma)),
    "SMA.str -> THL.str": OneToOne(SMA.str.V, THL.str.Isyn, np.ones(n_sma* n_ppc))


}
for name, gain in list(gains.items()):
    connections[name].gain = gain


def set_trial(task, trial=0):
    trg = task[trial]["target"]
    TARGET.str.Iext[trg] = 7

    return trg


def iterate(dt):
    # Flush connections
    for connection in list(connections.values()):
        connection.flush()

    # Propagate activities
    for connection in list(connections.values()):
        connection.propagate()

    # Compute new activities
    for structure in structures:
        structure.evaluate(dt)


def reset():
    # reset_weights()
    reset_activities()
    reset_history()


def reset_weights():
    connections["PPC.str -> SMA.str"].weights = 0.5 * Wppc2sma()
    connections["PPC.str -> SMA.str"].weights = 0.5 * Wppc2sma()

    # connections["M1.str -> M1.str"].weights = 0.01 * np.ones(n_m1*n_m1)
    # connections["M1.str -> M1.str"].weights = 0.01 * np.ones(n_m1*n_m1)

    connections["SMA.str -> STR_SMA_PPC.str"].weights = weights(n_sma * n_ppc)
    connections["SMA.str -> STR_SMA_PPC.str"].weights = weights(n_sma * n_ppc)
    connections["PPC.str -> STR_SMA_PPC.str"].weights = 0.5 * np.ones(n_sma * n_ppc)
    connections["PPC.str -> STR_SMA_PPC.str"].weights = 0.5 * np.ones(n_sma * n_ppc)


def reset_activities():
    for structure in structures:
        structure.reset()


def reset_arm1_activities():
    for structure in arm_structures:
        structure.str.U = 0
        structure.str.V = 0
        structure.str.Isyn = 0
        structure.str.Iext = 0

    for structure in BG_structures:
        structure.smath1.U = 0
        structure.smath1.V = 0
        structure.smath1.Isyn = 0
        structure.smath1.Iext = 0


def reset_arm2_activities():
    for structure in arm_structures:
        structure.str.U = 0
        structure.str.V = 0
        structure.str.Isyn = 0
        structure.str.Iext = 0

    for structure in BG_structures:
        structure.smath2.U = 0
        structure.smath2.V = 0
        structure.smath2.Isyn = 0
        structure.smath2.Iext = 0


def history():
    histor = np.zeros(duration, dtype=dtype)

    histor["PPC"]["str"] = PPC.str.history[:duration]
    histor["SMA"]["str"] = SMA.str.history[:duration]
    # histor["M1_in"]["str"] = M1_in.str.history[:duration]
    # histor["M1_out"]["str"] = M1_out.str.history[:duration]
    # histor["ISM"]["str"] = ISM.str.history[:duration]
    histor["TARGET"]["str"] = TARGET.str.history[:duration]
    histor["STR"]["str"] = STR.str.history[:duration]
    histor["STN"]["str"] = STN.str.history[:duration]
    histor["THL"]["str"] = THL.str.history[:duration]
    histor["GPI"]["str"] = GPI.str.history[:duration]
    return histor


def reset_history():

    PPC.str.history[:duration] = 0
    SMA.str.history[:duration] = 0
    # M1_in.str.history[:duration] = 0
    # M1_out.str.history[:duration] = 0
    # ISM.str.history[:duration] = 0
    TARGET.str.history[:duration] = 0
    STR.str.history[:duration] = 0
    THL.str.history[:duration] = 0


def M1_learning1(m1_th1, m1_th2, Wmax = 0.75, Wmin = 0.00):

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * M1.str.V[m1_th1]
    W = connections["M1.str -> M1.str"].weights
    W.reshape((n_m1, n_m1))[m1_th1, m1_th2] += dw * (Wmax - W.reshape((n_m1, n_m1))[m1_th1, m1_th2]) * (
        W.reshape((n_m1, n_m1))[m1_th1, m1_th2] - Wmin)
    connections["M1.str -> M1.str"].weights = W

def M1_learning2(m1_th1, m1_th2, Wmax = 0.75, Wmin = 0.00):
    dw = alpha_LTP_ctx * M1.str.V[m1_th2]
    W = connections["M1.str -> M1.str"].weights
    W.reshape((n_m1, n_m1))[m1_th2, m1_th1] += dw * (Wmax - W.reshape((n_m1, n_m1))[m1_th2, m1_th1]) * (
        W.reshape((n_m1, n_m1))[m1_th2, m1_th1] - Wmin)
    connections["M1.str -> M1.str"].weights = W


def SMA_learning1(reward, ppc, sma):
    # print "reward: ", reward
    # Compute prediction error
    error = reward - SMA_value_th1.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    SMA_value_th1.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # SMA
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.str.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["SMA.str -> STR_SMA_PPC.str"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * \
                                           (W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["SMA.str -> STR_SMA_PPC.str"].weights = W
    # print 'SMA1: %d   PPC1: %d  \nSMA->STR: ' % (sma, ppc), W.reshape((n_sma, n_ppc))[sma, ppc]

    # Compute prediction error
    error = reward - PPC_value_th1.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    PPC_value_th1.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # PPC
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.str.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["PPC.str -> STR_SMA_PPC.str"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.str -> STR_SMA_PPC.str"].weights = W
    # print 'PPC->STR: ', W.reshape((n_sma, n_ppc))[sma, ppc]

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.str.V[ppc]
    W = connections["PPC.str -> SMA.str"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.str -> SMA.str"].weights = W
    # print 'PPC->SMA: ', W.reshape((n_sma, n_ppc))[sma, ppc]


def SMA_learning2(reward, ppc, sma):
    # if arm_pos == target:
    #     reward = 1
    # else:
    #     reward = 0

    # print "reward: ", reward
    # Compute prediction error
    error = reward - SMA_value_th2.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    SMA_value_th2.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # SMA
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.str.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["SMA.str -> STR_SMA_PPC.str"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["SMA.str -> STR_SMA_PPC.str"].weights = W
    # print 'SMA2: %d   PPC2: %d  \nSMA->STR: ' % (sma, ppc), W.reshape((n_sma, n_ppc))[sma, ppc]

    # Compute prediction error
    error = reward - PPC_value_th2.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    PPC_value_th2.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # PPC
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.str.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["PPC.str -> STR_SMA_PPC.str"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.str -> STR_SMA_PPC.str"].weights = W
    # print 'PPC->STR: ', W.reshape((n_sma, n_ppc))[sma, ppc]

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.str.V[ppc]
    W = connections["PPC.str -> SMA.str"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.str -> SMA.str"].weights = W
    # print 'PPC->SMA: ', W.reshape((n_sma, n_ppc))[sma, ppc]


# def debug_arm(theta=1):
#     if theta == 1:
#         ppc = np.argmax(PPC.str.V)
#         sma = np.argmax(SMA.str.V)
#         arm = np.argmax(ARM.str.V)
#         m1 = np.argmax(M1.str.V)
#         mot = buttons[np.argmax(CTX.mot.V), 0]
#         print "Motor CTX: ", mot
#         # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
#         # print "SMA: ", sma
#         # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
#         print "Arm: ", arm
#         print
#     else:
#         ppc = np.argmax(PPC.str.V)
#         sma = np.argmax(SMA.str.V)
#         arm = np.argmax(ARM.str.V)
#         m1 = np.argmax(M1.str.V)
#         mot = buttons[np.argmax(CTX.mot.V), 1]
#         print "Motor CTX: ", mot
#         # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
#         # print "SMA: ", sma
#         # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
#         print "Arm: ", arm
#         print

def debug_arm():
        ppc = np.argmax(PPC.str.V)
        sma = np.argmax(SMA.str.V)
        arm1 = np.argmax(ARM.str.V)
        m1 = np.argmax(M1.str.V)
        mot1 = buttons[np.argmax(CTX.mot.V), 0]
        # print "Motor CTX: ", mot
        # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
        # print "SMA: ", sma
        # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
        # print "Arm: ", arm
        ppc = np.argmax(PPC.str.V)
        sma = np.argmax(SMA.str.V)
        arm2 = np.argmax(ARM.str.V)
        m1 = np.argmax(M1.str.V)
        mot2 = buttons[np.argmax(CTX.mot.V), 1]
        print("Motor CTX: ", mot1, mot2)
        # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
        # print "SMA: ", sma
        # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
        print("Arm: ", arm1,arm2)
        print()


def debug_arm_learning():
    print("  SMA Values	1		: ", SMA_value_th1)
    print("  PPC Values	1		: ", PPC_value_th1)
    print("  PPC -> SMA Weights 1: ", connections["PPC.str -> SMA.str"].weights.reshape(n_sma, n_ppc))
    print("  SMA -> STR Weights 1: ", connections["SMA.str -> STR_SMA_PPC.str"].weights.reshape(n_sma, n_ppc))
    print("  PPC -> STR Weights 1: ", connections["PPC.str -> STR_SMA_PPC.str"].weights.reshape(n_sma, n_ppc))
    print("  SMA Values	2		: ", SMA_value_th2)
    print("  PPC Values	2		: ", PPC_value_th2)
    print("  PPC -> SMA Weights 2: ", connections["PPC.str -> SMA.str"].weights.reshape(n_sma, n_ppc))
    print("  SMA -> STR Weights 2: ", connections["SMA.str -> STR_SMA_PPC.str"].weights.reshape(n_sma, n_ppc))
    print("  PPC -> STR Weights 2: ", connections["PPC.str -> STR_SMA_PPC.str"].weights.reshape(n_sma, n_ppc))
    print()
