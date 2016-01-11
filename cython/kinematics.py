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
    k, l = -1, 0
    for i in crd:
        k += 1
        l = 0
        for j in crd:
            coord[k, l, :] = i, j
            l += 1
    return coord.reshape((9,2))
def closer(init_pos, pos, target):
    d1 = distance(init_pos,target)
    d2 = distance(pos, target)
    return d1>d2



if __name__ == "__main__":

    coord = task_coordinations()
    print(coord.shape)
    print(closer(coord[0],coord[1],coord[6]))
    print(closer(coord[0],coord[3],coord[5]))
    print(closer(coord[2],coord[1],coord[5]))
    # for i in range(9):
    #     print(coord[i])
    # for goal in range(5):
    #     d = distance(coord[0],coord[i])
    #     for j in range(9):
    #         d2 = distance(coord[j], coord[i])
    #         print("Distance 0 --> %d: %f" % (i, d))
    #         print("Distance %d --> %d: %f" % (j, i, d2))
    #         print("Closer or Further: %f" % (d>d2))
    #         print("\n\n")
    # plt.plot(coord[:, 0], coord[:, 1], '*')
    # plt.xlim([-1.5, 1.5])
    # plt.ylim([-1.5, 1.5])
    # plt.show()