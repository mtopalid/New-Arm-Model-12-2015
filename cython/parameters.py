# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
import numpy as np
from kinematics import *

# Population size
n_targets = 9
n_ism = 9
n_sma = 5
n_arm = 9
n_m1_out = n_arm
n_m1_in = n_arm * n_sma
n_ppc = n_arm * n_targets
# Learning Positions
n_learning_positions_trials = 81*4#36  # 81*81*20
simulations = 100

coord = task_coordinations()

# buttons = np.ones((n, 2))
# buttons[0, :] = [4, 1]  # [90,75]
# buttons[1, :] = [1, 6]  # [75,100]
# buttons[2, :] = [4, 6]  # [90,100]
# buttons[3, :] = [6, 1]  # [100,75]


# --- Time ---
ms = 0.001
duration = int(3. / ms)
duration_learning_positions = int(16. / ms)
dt = 1 * ms
tau = 10 * ms

# --- Learning ---
a = 1.
alpha_CUE = 0.0025 * a  # 0.0005
alpha_LTP = 0.005 * a
alpha_LTD = 0.00375 * a
alpha_LTP_ctx = alpha_LTP ** 2 * a  # 0.000025

# --- Sigmoid ---
Vmin = 0
Vmax = 20
Vh = 16
Vc = 3

# --- Model ---
decision_threshold = 40
PPC_rest = 3.0
TARGET_rest = -3.0
a = 1. / 1.
SMA_rest = -3.0 * a
M1_in_rest = -20.0
M1_out_rest = -10.0
# ARM_rest = -30.0
STR_rest = 0.0
STN_rest = -10.0
GPE_rest = -10.0
GPI_rest = -10.0
THL_rest = -40.0 * a

# Noise level (%)
Cortex_N = 0.01
Striatum_N = 0.01
STN_N = 0.01
GPi_N = 0.01
GPe_N = 0.01
Thalamus_N = 0.01

# --- Cues & Rewards ---
Value_cue = 7
noise_cue = 0.001
rewards = np.zeros((n_targets, n_arm, n_arm))
for trg in range(n_targets):
    for initpos in range(n_arm):
        for pos in range(n_arm):
            if trg == initpos and trg == pos:
                rewards[trg, initpos, pos] = 1
            else:
                rewards[trg, initpos, pos] = closer(coord[initpos], coord[pos], coord[trg])
# rewards[np.where(rewards==0)]= -0.5
# -- Weight ---
Wmin = 0.25
Wmax = 0.75

gains = {

    "PPC -> PPC": +0.5,

    "SMA -> SMA": +0.5,

    "M1_in -> M1_in": +0.5,

    "M1_out -> M1_out": +0.5,

    "ISM -> ISM": +0.5,

    # Input
    "TARGET -> PPC": +1.0,  # +0.5,#

    "TARGET -> ISM": +0.0,  # +0.5,#
    # Input To SMA
    "PPC -> SMA": +0.5,  # +1.0,#

    # Input To M1in
    "SMA -> M1_in": +0.1,  # +1.0,#

    "M1_in -> M1_out": +0.1,
    "ISM -> M1_out": +0.0,## 35,#


    # SMA <-> BG

    "SMA -> STN": +0.1,
    "PPC -> STN": +0.,

    "SMA -> STR": +2.,  # +0.5,#
    "PPC -> STR": +2.,  # +0.5,#

    "STR -> GPE": -2.0,
    "STR -> GPI": -2.0,

    "GPE -> STN": -1.,

    "STN -> GPI": +0.1,

    "GPI -> THL": -0.2,

    "THL -> SMA": +0.4,
    "SMA -> THL": +0.1,

}

dtype = [
    ("PPC", [("str", float, n_ppc)]),
    ("SMA", [("str", float, n_sma)]),
    ("M1_in", [("str", float, n_m1_in)]),
    ("M1_out", [("str", float, n_m1_out)]),
    ("ISM", [("str", float, n_ism)]),
    ("STR", [("str", float, n_sma)]),
    ("STN", [("str", float, n_sma)]),
    ("THL", [("str", float, n_sma)]),
    ("GPI", [("str", float, n_sma)]),
    ("TARGET", [("str", float, n_targets)])]

Wm1in2mout = np.array([[-1, 3, 0, 1, -1],
                       [-1, 4, 1, 2, 0],
                       [-1, 5, 2, -1, 1],
                       [0, 6, 3, 4, -1],
                       [1, 7, 4, 5, 3],
                       [2, 8, 5, -1, 4],
                       [3, -1, 6, 7, -1],
                       [4, -1, 7, 8, 6],
                       [5, -1, 8, -1, 7]])
