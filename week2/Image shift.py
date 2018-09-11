import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

def imshift(x, k, l):
    height, width = x.shape
    res = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            res[h,w] = x[(h+k) % height, (w+l) % width]  # consider periodical boundary conditions
    return res


#Sample call and Plotting code
#“lake.png” and "windmill.png"

img1 = imread('lake.png')
img2 = imread('windmill.png')

img1 = imshift(img1, 100, -50)
img2 = imshift(img2, 100, -50)

plt.subplot(1, 2, 1)
plt.imshow(img1,cmap = plt.get_cmap('gray'))
plt.subplot(1, 2, 2)
plt.imshow(img2,cmap = plt.get_cmap('gray'))
plt.show()