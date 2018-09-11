import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


def sampling_quantization(img):
    height, width = img.shape
    width_sample, height_sample = width//10, height//10
    img_resample = np.zeros((height_sample, width_sample))

    # interpolation method
    # I use the average value as the pixel of of the original 10 * 10 area

    for h in range(height_sample):
        for w in range(width_sample):
            img_resample[h][w] = np.mean(img[h*10:(h+1)*10, w*10:(w+1)*10])

    img_res = img_resample//(256/5)
    return img_resample, img_res

#Import image here
img = imread('peppers.png')

#Sample call and Plotting code
img_resample, img_res = sampling_quantization(img)
plt.subplot(1, 2, 1)
plt.imshow(img_resample,cmap = plt.get_cmap('gray'))
plt.subplot(1, 2, 2)
plt.imshow(img_res, cmap = plt.get_cmap('gray'))
plt.show()
