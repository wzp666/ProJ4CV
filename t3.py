import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
#
# img = imread('castle.png')
# print(img.shape)
# print(img)
t1 = np.array([[52, -2, 15],
            [50, 25, 30],
            [235, 40, 45]])
t2 = np.array([[52, 0, 15],
            [0, 25, 30],
            [235, 40, 3]])


# t3 = np.append(t1, t2,axis=-1)
# t.append(t2)
# print(np.sort(t1,axis=-1))
# print(t1[1,2:3])
print(np.max(t1,t2, axis=-1))