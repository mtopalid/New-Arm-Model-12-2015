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
M1_in = Structure(tau=tau, rest=M1_in_rest, noise=Cortex_N, activation=clamp, n=n_m1_in)
M1_out = Structure(tau=tau, rest=M1_out_rest, noise=Cortex_N, activation=clamp, n=n_m1_out)
ISM = Structure(tau=tau, rest=-a, noise=Cortex_N, activation=clamp, n=n_ism)

# BG
STR = Structure(tau=tau, rest=STR_rest, noise=Striatum_N, activation=clamp, n=n_sma)
STN = Structure(tau=tau, rest=STN_rest, noise=STN_N, activation=clamp, n=n_sma)
GPE = Structure(tau=tau, rest=GPE_rest, noise=GPe_N, activation=clamp, n=n_sma)
GPI = Structure(tau=tau, rest=GPI_rest, noise=GPi_N, activation=clamp, n=n_sma)
THL = Structure(tau=tau, rest=THL_rest, noise=Thalamus_N, activation=clamp, n=n_sma)

structures = (PPC, SMA, M1_in, M1_out, ISM, TARGET, STR, STN, GPE, GPI, THL)  #

SMA_values = 0.5 * np.ones(n_sma * n_ppc)
PPC_values = 0.5 * np.ones(n_sma * n_ppc)


# Add noise to weights
def weights(shape, s=0.005, initial=0.5):
    N = np.random.normal(initial, s, shape)
    N = np.minimum(np.maximum(N, 0.0), 1.0)
    return Wmin + (Wmax - Wmin) * N


def Wlateral(n):
    return (2 * np.eye(n) - np.ones((n, n))).ravel()


def Wppc2sma(n1=n_sma, n2=n_arm, n3=n_targets):
    w = weights(n1 * n2 * n3, s=0.0005).reshape((n1, n2, n3))
    # w = np.ones((n1, n2, n3))
    w[0, :3, :] = 0
    w[1, -3:, :] = 0
    w[3, 2::3, :] = 0
    w[4, ::3, :] = 0

    return w.reshape(n1 * n2 * n3)


def WM1in2M1out(n1=n_arm, n2=n_sma, n3=n_m1_out):
    w = np.zeros((n3, n1, n2))
    for i in range(n3):
        w[i][np.where(Wm1in2mout == i)] = 1
    return w.reshape(n3 * n1 * n2)


# np.set_printoptions(threshold='nan')
# print(WM1in2M1out().reshape((n_m1_out, n_arm, n_sma)))
# print(Wppc2sma().reshape((n_sma, n_arm, n_targets)))
# Connectivity 
connections = {

    # Lateral connectivity

    "PPC -> PPC": AllToAll(PPC.str.V, PPC.str.Isyn, Wlateral(n_ppc)),

    "SMA -> SMA": AllToAll(SMA.str.V, SMA.str.Isyn, Wlateral(n_sma)),

    "M1_in -> M1_in": AllToAll(M1_in.str.V, M1_in.str.Isyn, Wlateral(n_m1_in)),

    "M1_out -> M1_out": AllToAll(M1_out.str.V, M1_out.str.Isyn, Wlateral(n_m1_out)),

    "ISM -> ISM": AllToAll(ISM.str.V, ISM.str.Isyn, Wlateral(n_ism)),

    # Input to PPC
    "TARGET -> PPC": OneToColumn(TARGET.str.V, PPC.str.Isyn, np.ones(n_targets * n_arm), np.array([n_arm, n_targets])),

    "TARGET -> ISM": AllToAll(TARGET.str.V, ISM.str.Isyn, weights(n_targets*n_arm, 0.0005)),

    # Input To SMA
    "PPC -> SMA": AllToAll(PPC.str.V, SMA.str.Isyn, Wppc2sma(), np.array([n_sma, n_arm, n_targets])),

    # Input To M1in
    "SMA -> M1_in": OneToColumn(SMA.str.V, M1_in.str.Isyn, np.ones(n_m1_in), np.array([n_arm, n_sma])),

    # "ISM -> M1_out": OneToOne(ISM.str.V, M1_out.str.Isyn, np.ones(n_arm)),

    "M1_in -> M1_out": AllToAll(M1_in.str.V, M1_out.str.Isyn, WM1in2M1out()),

    "ISM -> M1_out": OneToOne(ISM.str.V, M1_out.str.Isyn, np.ones(n_arm)),

    # BG
    "SMA -> STN": OneToOne(SMA.str.V, STN.str.Isyn, np.ones(n_sma * n_ppc)),
    "SMA -> STR": OneToOne(SMA.str.V, STR.str.Isyn, np.ones(n_sma * n_ppc)),
    # plastic (RL)
    "PPC -> STR": AllToAll(PPC.str.V, STR.str.Isyn, weights(n_sma * n_ppc)),
    "PPC -> STN": AllToAll(PPC.str.V, STN.str.Isyn, np.ones(n_sma * n_ppc)),
    # plastic (RL)

    "STR -> GPE": OneToOne(STR.str.V, GPE.str.Isyn, np.ones(n_sma * n_ppc)),
    "STR -> GPI": OneToOne(STR.str.V, GPI.str.Isyn, np.ones(n_sma * n_ppc)),
    # #
    "GPE -> STN": OneToOne(GPE.str.V, STN.str.Isyn, np.ones(n_sma * n_ppc), np.array([n_sma, n_ppc])),
    "STN -> GPI": AllToAll(STN.str.V, GPI.str.Isyn, np.ones(n_sma * n_sma * n_ppc)),

    "GPI -> THL": OneToOne(GPI.str.V, THL.str.Isyn, np.ones(n_sma)),
    "THL -> SMA": OneToOne(THL.str.V, SMA.str.Isyn, np.ones(n_sma)),
    "SMA -> THL": OneToOne(SMA.str.V, THL.str.Isyn, np.ones(n_sma * n_ppc))

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
    connections["PPC -> SMA"].weights = Wppc2sma()

    connections["PPC -> STR"].weights = weights(n_sma * n_ppc)


def reset_activities():
    for structure in structures:
        structure.reset()


def history(dur=duration):
    histor = np.zeros(dur, dtype=dtype)

    histor["PPC"]["str"] = PPC.str.history[:dur]
    histor["SMA"]["str"] = SMA.str.history[:dur]
    histor["M1_in"]["str"] = M1_in.str.history[:dur]
    histor["M1_out"]["str"] = M1_out.str.history[:dur]
    # histor["ISM"]["str"] = ISM.str.history[:dur]
    histor["TARGET"]["str"] = TARGET.str.history[:dur]
    histor["STR"]["str"] = STR.str.history[:dur]
    histor["STN"]["str"] = STN.str.history[:dur]
    histor["THL"]["str"] = THL.str.history[:dur]
    histor["GPI"]["str"] = GPI.str.history[:dur]
    return histor


def reset_history():
    PPC.str.history[:duration] = 0
    SMA.str.history[:duration] = 0
    M1_in.str.history[:duration] = 0
    M1_out.str.history[:duration] = 0
    # ISM.str.history[:duration] = 0
    TARGET.str.history[:duration] = 0
    STR.str.history[:duration] = 0
    THL.str.history[:duration] = 0


def SMA_learning(reward, ppc, sma):
    # Compute prediction error
    error = reward - PPC_values.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    PPC_values.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # Update weights
    lrate = alpha_LTP if error > 0 else alpha_LTD
    dw = error * lrate * STR.str.V[sma]
    W = connections["PPC -> STR"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw
    connections["PPC -> STR"].weights = W

    # # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.str.V[ppc]
    W = connections["PPC -> SMA"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC -> SMA"].weights = W
def ISM_learning(trg, ism):
    # Target -> ISM
    dw = alpha_LTP_ctx * TARGET.str.V[trg]
    W = connections["TARGET -> ISM"].weights
    W.reshape((n_arm, n_targets))[ism, trg] += dw * (Wmax - W.reshape((n_arm, n_targets))[ism, trg]) * (
        W.reshape((n_arm, n_targets))[ism, trg] - Wmin)
    connections["TARGET -> ISM"].weights = W
    print('Target: %d   ISM: %d' % (trg, ism))

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
    print("Arm: ", arm1, arm2)
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
