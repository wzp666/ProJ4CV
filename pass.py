import time
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


def imkernel(tau, s1, s2):
    w = lambda i, j: np.exp(-(i ** 2 + j ** 2) / (2 * tau ** 2))
    # normalization
    i, j = np.mgrid[-s1:s1, -s2:s2]
    Z = np.sum(w(i, j))
    nu = lambda i, j: w(i, j) / Z * (np.absolute(i) <= s1 & np.absolute(j) <= s2)
    return nu


# Create imconvolve_naive function,
def imconvolve_naive(im, nu, s1, s2):
    (n1, n2) = im.shape
    xconv = np.zeros((n1, n2))
    for h in range(s1,n1-s1):
        for w in range(s1,n2-s2):
            for x in range(-s1, s1+1):
                for y in range(-s2, s2+1):
                    xconv[h,w]+=nu(x,y)*im[h+x, w+y]
    return xconv



# Create imconvolve_spatial function
def imconvolve_spatial(im, nu, s1, s2):
    (n1, n2) = im.shape
    xconv = np.zeros((n1, n2))
    for h in range(n1):
        for w in range(n2):
            for x in range(-s1, s1+1):
                for y in range(-s2, s2+1):
                    xconv[h,w]+=nu(x,y)*im[(h+x)%n1, (w+y)%n2]
    return xconv


def imshift(x, k, l):
    h,w = x.shape
    xshifted = np.zeros((h,w))
    if k == 0:
        k = h
    if l == 0:
        l = w
    xshifted[-k:,-l:] = x[:k,:l]
    xshifted[:-k,:-l] = x[k:,l:]
    xshifted[:-k,-l:] = x[k:,:l]
    xshifted[-k:,:-l] = x[:k,l:]
    return xshifted


def test_imconvolve():
    tau = 1
    s1 = 1
    s2 = 1
    img = imread('windmill.png')
    t0 = time.time()
    img_con = imconvolve_naive(img, imkernel(tau, s1, s2), s1, s2)
    t1 = time.time()
    img_spa = imconvolve_spatial(img, imkernel(tau, s1,s2), s1, s2)
    t2 = time.time()

    T1 =str(float('%.2f' % (t1-t0)))
    T2 = str(float('%.2f' % (t2-t1)))
    plt.subplot(1, 3, 1)
    plt.title("original image")
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.subplot(1, 3, 2)
    plt.title("cost time:"+T1)
    plt.imshow(img_con, cmap=plt.get_cmap('gray'))
    plt.subplot(1, 3, 3)
    plt.title("cost time:"+T2)
    plt.imshow(img_spa, cmap=plt.get_cmap('gray'))
    plt.show()
# Sample call and Plotting code

test_imconvolve()
