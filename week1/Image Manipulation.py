import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


# Rotate image (img) by 90 anticlockwise
def rotate90(array):
    sp = array.shape
    tem = np.array([[[0, 0, 0]] * sp[0]] * sp[1])
    for x in range(sp[0]):
        for y in range(sp[1]):
            tem[sp[1] - y - 1][x] = array[x][y]
    return tem


# Roate image (img) by an angle (ang) in anticlockwise direction
# Angle is assumed to be divisible by 90 but may be negative
def rotate(img, ang=0):
    assert ang % 90 == 0
    ang = ang % 360
    while ang < 0:
        ang += 90
    while ang != 0:
        img = rotate90(img)
        ang -= 90
    return img


# Import image here
img1 = imread('pepsi.jpg')

# Sample call
img90 = rotate(img1, 90)
img180 = rotate(img1, 180)
img270 = rotate(img1, 270)
img360 = rotate(img1, 360)
# Plotting code below

plt.subplot(2, 2, 1)  # first plot
plt.imshow(img90)
plt.subplot(2, 2, 2)  # second plot
plt.imshow(img180)
plt.subplot(2, 2, 3)  # third plot
plt.imshow(img270)
plt.subplot(2, 2, 4)  # fourth plot
plt.imshow(img360)
plt.show()
