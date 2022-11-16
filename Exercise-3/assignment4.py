#!/usr/bin/env python

"""Exercise 3"""

# 3rd party libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np

__author__ = "Marcus Jan Almert, Luigi Vinzenz Portwich, Vladimir Dimitrov Spassov"
# M:119915, L:119649, V:119606

# setting the figure size
plt.rcParams["figure.figsize"] = (40, 30)
np.set_printoptions(precision=3)


def main():
    """Main function"""
    # ------------------------------ Task 1 ------------------------------ #
    # widht = 1024
    # heigth = 753
    input_img = cv2.imread("Image_without_digital_watermark.jpg")
    # change type from uint8 to float and normalize values between [0,1]
    input_imgf_normalized = input_img.astype('float32') / 255.0
    # if the image can not be split into 8x8 blocks perfectly, remove the remaining pixels
    input_imgf_normalized = cv2.resize(input_imgf_normalized, (input_img.shape[1] - (input_img.shape[1] % 8),
                                                               input_img.shape[0] - (input_img.shape[0] % 8)))

    # Define the number of plots here
    fig = plt.figure()

    # dct output destination
    dct_img = np.zeros(shape=input_img.shape, dtype=np.float32)

    for x in range(0, input_imgf_normalized.shape[0] - 8, 8):
        for y in range(0, input_imgf_normalized.shape[1] - 8, 8):
            for rgb_channel in range(0, 3):
                dct_img[x:x + 8, y:y + 8, rgb_channel] = cv2.dct(input_imgf_normalized[x:x + 8, y:y + 8, rgb_channel])

    output_multiply = dct_img[0:8, 0:8, 0]
    output_add = dct_img

    # Name plot
    plt.suptitle(
        "\nImage Analysis and Object Recognition: Assignment 4\n\n"
        "Marcus Almert, Luigi Portwich, Vladimir Spassov\n",
        fontsize=45)

    # plot images
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(input_img)
    ax1.set_title("Original Image", fontsize="30")

    fig.tight_layout()
    fig.show()
    fig.savefig("output.pdf")


if __name__ == "__main__":
    main()
