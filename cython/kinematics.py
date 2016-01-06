import numpy as np
import matplotlib.pyplot as plt


# Given the angles of the joints returns position
def coordinations(theta1, theta2, l1=1., l2=1., pos0=np.array([0.0, 0.0]).reshape(1, 2)):
    x1 = np.cos(theta1) * l1 + pos0[0][0]
    y1 = np.sin(theta1) * l1 + pos0[0][1]
    x2 = np.cos(theta1 + theta2) * l2 + x1
    y2 = np.sin(theta1 + theta2) * l2 + y1

    return np.array([x2, y2]).reshape(1, 2)


# Given the angles of the joints returns muscles activation pattern of the arm
def muscles(theta1, theta2):
    g1, g3 = 0., 0.
    g2 = g1 - theta1 / np.pi
    g4 = g3 - theta2 / np.pi

    # g1 = theta1/np.pi/2.
    # g3 = theta2/np.pi/2.
    # g2, g4 = -g1, -g3

    g = np.array([g1, g2, g3, g4]).reshape((4, 1))
    return g


# Given position return the angles of the joints
def inverse(pos, l1=1., l2=1., pos0=np.array([0.0, 0.0]).reshape(1, 2)):
    pos[0][0] = np.array(pos[0][0]) - pos0[0][0]
    pos[0][1] = np.array(pos[0][1]) - pos0[0][1]

    # Elbow joint
    c2 = (pos[0][0] * pos[0][0] + pos[0][1] * pos[0][1] - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    s2 = np.sqrt(1. - c2 * c2)
    theta2 = np.arctan2(s2, c2)

    # Shoulder joint
    k1 = l1 + l2 * c2
    k2 = l2 * s2
    theta1 = np.arctan2(pos[0][1], pos[0][0]) - np.arctan2(k2, k1)

    return theta1, theta2


def tar_pos(whichTar):  # Target):
    # whichTar = np.where(Target != 0)[0][0]

    if whichTar == 0:
        pos = np.array([-1., 1.25]).reshape(1, 2)
    elif whichTar == 1:
        pos = np.array([-0.75, 1.]).reshape(1, 2)
    elif whichTar == 2:
        pos = np.array([-1., 0.75]).reshape(1, 2)
    else:
        pos = np.array([-1.25, 1.]).reshape(1, 2)

    # theta1, theta2 = inverse(pos)
    # musc = muscles(theta1, theta2)
    return pos  # , musc


def convert_rad2degr(radians):
    degrees = radians * 180 / np.pi
    return degrees


def conver_degr2rad(degrees):
    radians = degrees * np.pi / 180
    return radians


def distance(pos1, pos2):
    d = np.linalg.norm(np.array(pos1) - np.array(pos2))
    return d


def task_coordinations():
    crd = np.linspace(-1, 1, 2 / 1. + 1)
    coord = np.zeros([crd.size, crd.size, 2])
    # angles = np.zeros([crd.size, crd.size, 2])
    # musc = np.zeros([crd.size, crd.size, 4])
    k, l = -1, 0
    for i in crd:
        k += 1
        l = 0
        for j in crd:
            coord[k, l, :] = i, j
            # angles[k, l, :] = inverse(coord[k, l, :].reshape(1, 2))
            # musc[k, l, :] = muscles(angles[k, l, 0], angles[k, l, 1]).reshape((4,))
            l += 1
    return coord.reshape((9,2))

if __name__ == "__main__":
    # initial position of the arm
    # pos_f = np.array([-1., 1.]).reshape(1, 2)
    # th1_f, th2_f = inverse(pos_f)
    # Arm_f = muscles(th1_f, th2_f)
    # print convert_rad2degr(th1_f)
    # print distance([0, 0], [0, 1])

    coord = task_coordinations()
    print(coord.shape)
    for i in range(9):
        print(coord[i])

    # plt.plot(musc[:,:,0],musc[:,:,1], '*')
    # plt.plot(musc[:,:,0], 'bo')
    # plt.plot(musc[:,:,1], 'go')
    # plt.plot(musc[:,:,2], 'ro')
    # plt.plot(musc[:,:,3], 'mo')
    plt.plot(coord[:, 0], coord[:, 1], '*')
    # plt.plot(coord[4,1,0],coord[4,1,1], 'bo')
    # plt.plot(coord[1,6,0],coord[1,6,1], 'bo')
    # plt.plot(coord[4,6,0],coord[4,6,1], 'bo')
    # plt.plot(coord[6,1,0],coord[6,1,1], 'bo')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.show()
    # for i in range(crd.size):
    #     print((coord[i]))
    #     print((musc[i]))
    #     print((convert_rad2degr(angles[i])))
    #     print('')
