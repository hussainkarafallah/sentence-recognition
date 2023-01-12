import cv2 as cv
import numpy as np
import random


def gaussian_blur(img, kernel_size=(3, 3), std=1):
    return cv.GaussianBlur(img, ksize=kernel_size, sigmaX=std)


def gaussian_noise(img, std=1, mn=0, mx=1):
    # adds the noise
    rng = np.random.default_rng()
    noise = rng.standard_normal(img.shape) * std
    new_img = img + noise

    # clip the values
    new_img = np.where(new_img < mn, mn, new_img)
    new_img = np.where(new_img > mx, mx, new_img)

    return new_img

def rect_mask(img, w, h, mx=1, num_masks=1):
    rect = np.full((h, w), mx)
    new_img = img.copy()
    for _ in range(num_masks):
        x = random.randint(0, img.shape[1]-w)
        y = random.randint(0, img.shape[0]-h)
        new_img[y:y+h, x:x+w] = rect

    return new_img


def change_light(img, ratio, mn=0, mx=1):
    new_img = img.copy()
    new_img = new_img*ratio

    # clip values
    new_img = np.where(new_img<mn, mn, new_img)
    new_img = np.where(new_img>mx, mx, new_img)

    return new_img
