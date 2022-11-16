#!/usr/bin/env python

"""Resources for Image Analysis Assignment 3"""

__author__ = "Marcus Jan Almert, Luigi Vinzenz Portwich, Vladimir Dimitrov Spassov"

# M:119915, L:119649, V:119606
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizing a greyscale image.
    @param image: greyscale image
    @return: normalized greyscale image [0.0, 1.0]
    """
    image = image / np.max(image)
    return image


def add_gaussian_noise(image: np.ndarray, mean, standard_var: float) -> np.ndarray:
    """
    Adds some gaussian noise to a given image.
    @param image: given image
    @param mean: mean
    @param standard_var: standard deviation
    @return: modified image
    """
    new_image = np.copy(image)
    # get dimensions of image
    row, col = new_image.shape
    # draws random samples from normal (gaussian) distribution, output in given shape
    gauss_noise = np.random.normal(mean, standard_var, (row, col))
    new_image = new_image + gauss_noise
    return new_image

