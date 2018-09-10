import random
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import time


def GaussianNoise(x, means, sigma):
    n1, n2 = x.shape
    noise_img = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            noise_img[i, j] = x[i, j] + random.gauss(means, sigma)
            if noise_img[i, j] < 0:
                noise_img[i, j] = 0
            elif noise_img[i, j] > 255:
                noise_img[i, j] = 255
    return noise_img


def test_imbilateral():
    sigma = 10
    H = 1
    s1 = s2 = 10
    img = imread('castle.png')

    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.axis('off')
    plt.imshow(img, cmap=plt.get_cmap('gray'))

    img_noise = GaussianNoise(img, 0, 10)
    plt.subplot(1, 3, 2)
    plt.title("noise")
    plt.axis('off')
    plt.imshow(img_noise, cmap=plt.get_cmap('gray'))

    t0 = time.time()
    img1 = imbilateral(img_noise, s1, s2, sigma, H)
    print("time cost:",time.time()-t0)
    plt.subplot(1, 3, 3)
    plt.title("bilateral")
    plt.axis('off')
    plt.imshow(img1, cmap=plt.get_cmap('gray'))
    plt.show()


def imbialteral_navie(img, s1, s2, sigma, H):
    n1, n2 = img.shape
    phi = lambda a: np.exp(-(max(a - 2 * sigma ** 2, 0) / (16 * H * sigma ** 2)))
    z = np.zeros((n1, n2))
    x = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    if (i+k)>(n1-1) or (j+l)>(n2-1):
                        continue
                    x[i, j] += phi((img[(i + k), (j + l)] - img[i, j]) ** 2) * img[(i + k), (j + l)]
                    z[i, j] += phi((img[(i + k), (j + l)] - img[i, j]) ** 2)
    x = x / z
    return x


def imbilateral(img, s1, s2, sigma, H):
    n1, n2 = img.shape
    phi = lambda a: np.exp(-(np.maximum(a - 2 * sigma ** 2, 0) / (16 * H * sigma ** 2)))
    z = np.zeros((n1, n2))
    x = np.zeros((n1, n2))
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            xshift = imshift(img, -k, -l)
            x += phi((xshift - img) ** 2) * xshift
            z += phi((xshift - img) ** 2)
    x = x / z
    return x


def imshift(x, k, l):
    h, w = x.shape
    xshifted = np.zeros((h, w))
    if k == 0:
        k = h
    if l == 0:
        l = w
    xshifted[-k:, -l:] = x[:k, :l]
    xshifted[:-k, :-l] = x[k:, l:]
    xshifted[:-k, -l:] = x[k:, :l]
    xshifted[-k:, :-l] = x[:k, l:]
    return xshifted

test_imbilateral()

