#!/usr/bin/env python

"""Exercise 3"""

# 3rd party libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np

__author__ = "Marcus Jan Almert, Luigi Vinzenz Portwich, Vladimir Dimitrov Spassov"
# M:119915, L:119649, V:119606

# setting the figure size
plt.rcParams["figure.figsize"] = (60, 60)
np.set_printoptions(precision=3)


def main():
    """Main function"""
    # ------------------------------ Task 1 ------------------------------ #
    # widht = 1024
    # heigth = 753
    input_img = cv2.imread("Image_without_digital_watermark.jpg")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # change type from uint8 to float and normalize values between [0,1]

    # if the image can not be split into 8x8 blocks perfectly, remove the remaining pixels
    height = input_img.shape[1] - (input_img.shape[1] % 8)
    width = input_img.shape[0] - (input_img.shape[0] % 8)
    input_imgf_normalized = input_img[0:width, 0:height]

    input_imgf_normalized = input_imgf_normalized.astype('float32') / 255.0

    # dct output destination
    dct_img = np.zeros(shape=input_imgf_normalized.shape, dtype=np.float32)

    multiplication_constants = [x for x in range(-5, 6) if x != 0]
    addition_constants = [x for x in range(-250, 300, 50) if x != 0]

    # precomputed dct image
    dct_img[:, :, 0] = cv2.dct(input_imgf_normalized[:, :, 0])
    dct_img[:, :, 1] = cv2.dct(input_imgf_normalized[:, :, 1])
    dct_img[:, :, 2] = cv2.dct(input_imgf_normalized[:, :, 2])

    #dct_img[4:8, 0:4, :] += 50


    #for x in range(0, input_imgf_normalized.shape[0] - 8, 8):
    #    for y in range(0, input_imgf_normalized.shape[1] - 8, 8):
    #        for rgb_channel in range(0, 3):
    #            dct_img[x:x + 8, y:y + 8, rgb_channel] = cv2.dct(input_imgf_normalized[x:x + 8, y:y + 8, rgb_channel])


    # create the plots
    fig1 = plt.figure()
    fig2 = plt.figure()

    # Name plot
    plt.suptitle(
        "\nImage Analysis and Object Recognition: Assignment 4\n\n"
        "Marcus Almert, Luigi Portwich, Vladimir Spassov\n",
        fontsize=45)

    # for color_channel in range(0,3):
    # x = 123
    # ausschnitt = dct_img[0:int(width/2), 0:int(height/2)]
    # idct_ausschnitt = np.zeros(shape=ausschnitt.shape, dtype=np.float32)
    # # idct_ausschnitt[:, :, 0] = cv2.idct(ausschnitt[:, :, 0])
    # idct_ausschnitt[:, :, 0] = cv2.idct(ausschnitt[:, :, 0])
    # idct_ausschnitt[:, :, 1] = cv2.idct(ausschnitt[:, :, 1])
    # idct_ausschnitt[:, :, 2] = cv2.idct(ausschnitt[:, :, 2])

    idct = np.zeros(shape=input_imgf_normalized.shape, dtype=np.float32)
    idct[:, :, 0] = cv2.idct(dct_img[:, :, 0])
    idct[:, :, 1] = cv2.idct(dct_img[:, :, 1])
    idct[:, :, 2] = cv2.idct(dct_img[:, :, 2])
    idct = (idct * 255)
    idct = cv2.cvtColor(idct, cv2.COLOR_BGR2RGB)
    cv2.imwrite("ausschnitt2.jpg", idct)


    for i, constant in enumerate(multiplication_constants):
        multiplication_dct_quadrant1_img = \
            multiplication_dct_quadrant2_img = \
            multiplication_dct_quadrant3_img = \
            multiplication_dct_quadrant4_img = dct_img
        multiplication_idct_quadrant1_img = \
            multiplication_idct_quadrant2_img = \
            multiplication_idct_quadrant3_img = \
            multiplication_idct_quadrant4_img = np.zeros(shape=input_imgf_normalized.shape, dtype=np.float32)
        for x_origin in range(0, input_imgf_normalized.shape[0] - 8, 8):
            for y_origin in range(0, input_imgf_normalized.shape[1] - 8, 8):
                for rgb_channel in range(0, 3):

                    # quadrant 1 manipulation
                    for x_window in range(4, 8):
                        for y_window in range(0, 4):
                            multiplication_dct_quadrant1_img[x_window, y_window, rgb_channel] = constant * \
                                                                                                multiplication_dct_quadrant1_img[
                                                                                                    x_origin + x_window,
                                                                                                    y_origin + y_window,
                                                                                                    rgb_channel]

                    # quadrant 2 manipulation
                    for x_window in range(4, 8):
                        for y_window in range(4, 8):
                            multiplication_dct_quadrant2_img[x_window, y_window, rgb_channel] = constant * \
                                                                                                multiplication_dct_quadrant2_img[
                                                                                                    x_origin + x_window,
                                                                                                    y_origin + y_window,
                                                                                                    rgb_channel]

                    # quadrant 3 manipulation
                    for x_window in range(0, 4):
                        for y_window in range(4, 8):
                            multiplication_dct_quadrant3_img[x_window, y_window, rgb_channel] = constant * \
                                                                                                multiplication_dct_quadrant3_img[
                                                                                                    x_origin + x_window,
                                                                                                    y_origin + y_window,
                                                                                                    rgb_channel]

                    # quadrant 4 manipulation
                    for x_window in range(0, 4):
                        for y_window in range(0, 4):
                            multiplication_dct_quadrant4_img[x_window, y_window, rgb_channel] = constant * \
                                                                                                multiplication_dct_quadrant4_img[
                                                                                                    x_origin + x_window,
                                                                                                    y_origin + y_window,
                                                                                                    rgb_channel]

                    # inverse dct
                    #multiplication_idct_quadrant1_img[x_origin:x_origin + 8, y_origin:y_origin + 8,
                    #rgb_channel] = cv2.idct(
                    #    multiplication_dct_quadrant1_img[x_origin:x_origin + 8, y_origin:y_origin + 8, rgb_channel])
                    #multiplication_idct_quadrant2_img[x_origin:x_origin + 8, y_origin:y_origin + 8,
                    #rgb_channel] = cv2.idct(
                    #    multiplication_dct_quadrant2_img[x_origin:x_origin + 8, y_origin:y_origin + 8, rgb_channel])
                    #multiplication_idct_quadrant3_img[x_origin:x_origin + 8, y_origin:y_origin + 8,
                    #rgb_channel] = cv2.idct(
                    #    multiplication_dct_quadrant3_img[x_origin:x_origin + 8, y_origin:y_origin + 8, rgb_channel])
                    #multiplication_idct_quadrant4_img[x_origin:x_origin + 8, y_origin:y_origin + 8,
                    #rgb_channel] = cv2.idct(
                    #    multiplication_dct_quadrant4_img[x_origin:x_origin + 8, y_origin:y_origin + 8, rgb_channel])

        # inverse dct for every colour channel
        multiplication_idct_quadrant1_img[:, :, 0] = cv2.idct(multiplication_dct_quadrant1_img[:, :, 0])
        multiplication_idct_quadrant2_img[:, :, 0] = cv2.idct(multiplication_dct_quadrant2_img[:, :, 0])
        multiplication_idct_quadrant3_img[:, :, 0] = cv2.idct(multiplication_dct_quadrant3_img[:, :, 0])
        multiplication_idct_quadrant4_img[:, :, 0] = cv2.idct(multiplication_dct_quadrant4_img[:, :, 0])

        multiplication_idct_quadrant1_img[:, :, 1] = cv2.idct(multiplication_dct_quadrant1_img[:, :, 1])
        multiplication_idct_quadrant2_img[:, :, 1] = cv2.idct(multiplication_dct_quadrant2_img[:, :, 1])
        multiplication_idct_quadrant3_img[:, :, 1] = cv2.idct(multiplication_dct_quadrant3_img[:, :, 1])
        multiplication_idct_quadrant4_img[:, :, 1] = cv2.idct(multiplication_dct_quadrant4_img[:, :, 1])

        multiplication_idct_quadrant1_img[:, :, 2] = cv2.idct(multiplication_dct_quadrant1_img[:, :, 2])
        multiplication_idct_quadrant2_img[:, :, 2] = cv2.idct(multiplication_dct_quadrant2_img[:, :, 2])
        multiplication_idct_quadrant3_img[:, :, 2] = cv2.idct(multiplication_dct_quadrant3_img[:, :, 2])
        multiplication_idct_quadrant4_img[:, :, 2] = cv2.idct(multiplication_dct_quadrant4_img[:, :, 2])

        # convert images from normalized floats to rgb uint8
        multiplication_idct_quadrant1_img = (multiplication_idct_quadrant1_img * 255).astype(np.uint8)
        multiplication_idct_quadrant2_img = (multiplication_idct_quadrant2_img * 255).astype(np.uint8)
        multiplication_idct_quadrant3_img = (multiplication_idct_quadrant3_img * 255).astype(np.uint8)
        multiplication_idct_quadrant4_img = (multiplication_idct_quadrant4_img * 255).astype(np.uint8)

        # plot images
        ax1 = fig1.add_subplot(10, 5, i * 5 + 1)
        ax1.imshow(multiplication_idct_quadrant1_img)
        ax1.set_title("Quadrant 1, Multiplication: " + str(constant), fontsize="30")
        ax1 = fig1.add_subplot(10, 5, i * 5 + 2)
        ax1.imshow(multiplication_idct_quadrant2_img)
        ax1.set_title("Quadrant 2, Multiplication: " + str(constant), fontsize="30")
        ax1 = fig1.add_subplot(10, 5, i * 5 + 3)
        ax1.imshow(input_img)
        ax1.set_title("Original Image", fontsize="30")
        ax1 = fig1.add_subplot(10, 5, i * 5 + 4)
        ax1.imshow(multiplication_idct_quadrant3_img)
        ax1.set_title("Quadrant 3, Multiplication: " + str(constant), fontsize="30")
        ax1 = fig1.add_subplot(10, 5, i * 5 + 5)
        ax1.imshow(multiplication_idct_quadrant4_img)
        ax1.set_title("Quadrant 4, Multiplication: " + str(constant), fontsize="30")

    # fig.tight_layout()
    fig1.show()
    fig1.savefig("output.pdf", dpi=100)


if __name__ == "__main__":
    main()
