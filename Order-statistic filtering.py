import random
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


def imstack(img, s1, s2):
    n1, n2 = img.shape
    s = (2*s1+1)*(2*s2+1)
    xstack = np.empty((n1, n2, 0))
    for k in range(-s1,s1+1):
        for l in range(-s2,s2+1):
            xshift = np.expand_dims(imshift(img, -k, -l), axis=-1)
            xstack = np.append(xshift,xstack, axis=-1)
    return xstack


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


def sort(simg):
    n1, n2 = simg.shape
    for h in range(n1):
        for w in range(n2):
            simg[h, w].sort()
    return simg


def imosf(x, typ, s1, s2):
    n1, n2 = x.shape
    xosf = np.zeros((n1, n2))
    xstack = imstack(x, s1, s2)
    if typ == "median":
        xosf = np.median(xstack, axis=-1)
    elif typ =="erode":
        xosf = np.min(xstack, axis=-1)
    elif typ == "dilate":
        xosf = np.max(xstack, axis=-1)
    elif typ == "trimmed":
        xosf = np.sort(xstack, axis=-1)
        s = 2*(s1+1)*2*(s2+1)
        xosf = np.mean(xosf[:, :, int(s*0.25):int(s*0.75)], axis=-1)
    elif typ == "close":
        tem = np.max(xstack, axis=-1)
        tem = imstack(tem, s1, s2)
        xosf =  np.min(tem, axis=-1)
    elif typ == "open":
        tem = np.min(xstack, axis=-1)
        tem = imstack(tem, s1, s2)
        xosf =  np.max(tem, axis=-1)
    return xosf


def pepper(x, rate):
    n1, n2 = x.shape
    noise_p = int(n1*n2*rate)
    for r in range(noise_p):
        x[random.randint(0, n1-1),random.randint(0, n2-1)] = random.choice([0, 255])
    return x


s1 = 2
s2 = 2
img = imread('castle.png')
img_noise = pepper(img, 0.05)

plt.subplot(1, 5, 1)
plt.title("noise")
plt.axis('off')
plt.imshow(img_noise ,cmap = plt.get_cmap('gray'))

img1 = imosf(img_noise,"median",s1,s2)
plt.subplot(1, 5, 2)
plt.title("median")
plt.axis('off')
plt.imshow(img1 ,cmap = plt.get_cmap('gray'))

img2 = imosf(img_noise,"trimmed",s1,s2)
plt.subplot(1, 5, 3)
plt.title("trimmed")
plt.axis('off')
plt.imshow(img2 ,cmap = plt.get_cmap('gray'))


img11 = imosf(img_noise,"close",s1,s2)
plt.subplot(1, 5, 4)
plt.title("closing")
plt.axis("off")
plt.imshow(img11 ,cmap = plt.get_cmap('gray'))

img22 = imosf(img_noise,"open",s1,s2)
plt.subplot(1, 5, 5)
plt.title("opening")
plt.axis("off")
plt.imshow(img22 ,cmap = plt.get_cmap('gray'))

plt.show()
