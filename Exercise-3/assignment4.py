#!/usr/bin/env python

"""Exercise 3"""

# 3rd party libraries
import cv2
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt
import numpy as np

# own stuff
from ressources import normalize_image, add_gaussian_noise

__author__ = "Marcus Jan Almert, Luigi Vinzenz Portwich, Vladimir Dimitrov Spassov"
# M:119915, L:119649, V:119606

# setting the figure size
plt.rcParams["figure.figsize"] = (40, 30)
np.set_printoptions(precision=3)


def calc_gog_kernels(standard_deviation):
    """return Gradient of Gaussian Matrices Gx and Gy"""
    # initialize list (we append to list for now and reshape later)
    Gx = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            Gx.append((-x / (2 * math.pi * pow(standard_deviation, 4))) * math.exp(
                -(pow(x, 2) + pow(y, 2)) / (2 * pow(standard_deviation, 2))))
    # reshape list to matrix column-wise
    Gx = np.reshape(Gx, (5, 5), order="F")
    return Gx, Gx.transpose()


def convolute(Gx, Gy, img):
    """return convolutions with kernels Gx and Gy to image img"""
    Ix = cv2.filter2D(img, ddepth=-1, kernel=Gx)
    Iy = cv2.filter2D(img, ddepth=-1, kernel=Gy)
    return Ix, Iy


def gradient_magnitude(Ix, Iy):
    """return gradient magnitude"""
    return np.sqrt(pow(Ix, 2) + pow(Iy, 2))


def gen_cluster_centers(cluster_count: int) -> list:
    result = []
    for i in range(cluster_count):
        result.append(list(np.random.choice(range(256), size=3)))
    return result


def k_mean_clustering(image, cluster_count):
    cluster_centers = gen_cluster_centers(cluster_count)
    clusters = [[] for i in range(cluster_count)]

    for x in range(len(image)):
        for y in range(len(image[0])):
            smallest_c_index = -1
            smallest_distance = 442  # max distance
            for c in range(len(cluster_centers)):
                current_distance = np.linalg.norm(np.array(cluster_centers[c]) - np.array(image[x][y]))
                if (current_distance < smallest_distance):
                    smallest_distance = current_distance
                    smallest_c_index = c
            clusters[smallest_c_index].append([x, y])

    segmented_image = np.full_like(image, 0)
    for cluster in clusters:
        temp = [image[z[0]][z[1]] for z in cluster]
        temp1_mean = np.mean(temp, axis=0)
        for point in cluster:
            segmented_image[point[0]][point[1]] = temp1_mean
    return segmented_image / 255.0


def eight_neighbors(point):
    """return the eight adjacent neighbors of a point, boundary violations are not handled"""
    # points start in upper left hand corner, continues clockwise
    return [[point[0]-1, point[1]-1], [point[0], point[1]-1], [point[0]+1, point[1]-1],
            [point[0]-1, point[1]], [point[0]+1, point[1]],
            [point[0]-1, point[1]+1], [point[0], point[1]+1], [point[0]+1, point[1]+1]]


def watershed(seeds, img):
    """region growing algorithm for a list of seeds"""

    # copy to prevent altering the original list
    seeds_ = copy.deepcopy(seeds)

    # out going image
    img_out = np.zeros(shape=img.shape, dtype=np.uint8)
    labels = np.zeros(shape=img.shape, dtype=np.uint8)

    # setting initial lables
    for i, seed in enumerate(seeds_):
        seed.reverse()
        labels[seed[0]][seed[1]] = i+1

    # list to track which points were
    processed = []

    while len(seeds_):
        seed = seeds_.pop(0)

        processed.append(seed)
        img_out[seed[0]][seed[1]] = 1

        neighbors = eight_neighbors(seed)

        for neighbor in neighbors:
            # test if the neighbor is in the image bounds as well if it was already processed
            if 0 <= neighbor[0] < img.shape[0] and 0 <= neighbor[1] < img.shape[1] and (neighbor not in processed):
                # checking the similarity of the gray values
                if np.abs(img[neighbor[0]][neighbor[1]] - img[seed[0]][seed[1]]) < 1e-04:
                    seeds_.append(neighbor)
                    processed.append(neighbor)
                    img_out[neighbor[0]][neighbor[1]] = 1
                    labels[neighbor[0]][neighbor[1]] = label_in_neighborhood(neighbor, labels)
    return img_out, labels


def label_in_neighborhood(point, labels):
    found_labels = []

    for neighbor in eight_neighbors(point):
        if 0 <= neighbor[0] < labels.shape[0] and 0 <= neighbor[1] < labels.shape[1]:
            if labels[neighbor[0]][neighbor[1]] != 0:
                found_labels.append(labels[neighbor[0]][neighbor[1]])

    if found_labels:
        return max(found_labels)
    return 0


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

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(task1_segmented_image_1)
    ax2.set_title("Segmented Image 1", fontsize="30")

    ax3 = fig.add_subplot(3, 3, 3, projection="3d")
    ax3.set_title("Feature Space 1", fontsize="30")
    R = []
    B = []
    G = []
    C = []
    for x in range(len(task_1_image_1)):
        for y in range(len(task_1_image_1[0])):
            R.append(task_1_image_1[x][y][0])
            G.append(task_1_image_1[x][y][1])
            B.append(task_1_image_1[x][y][2])
            C.append(task1_segmented_image_1[x][y])
    ax3.scatter(R, G, B, c=C)

    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(task_1_image_2)
    ax4.set_title("Original 2", fontsize="30")

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(task1_segmented_image_2)
    ax5.set_title("Segmented Image 2", fontsize="30")

    ax6 = fig.add_subplot(3, 3, 6, projection="3d")
    ax6.set_title("Feature Space 2", fontsize="30")
    R = []
    B = []
    G = []
    C = []
    for x in range(len(task_1_image_2)):
        for y in range(len(task_1_image_2[0])):
            R.append(task_1_image_2[x][y][0])
            G.append(task_1_image_2[x][y][1])
            B.append(task_1_image_2[x][y][2])
            C.append(task1_segmented_image_2[x][y])
    ax6.scatter(R, G, B, c=C)

    # Show color picture
    ax7 = fig.add_subplot(3, 3, 9)
    ax7.imshow(labels, cmap="tab20b")
    ax7.set_title("Watershed Segmentations", fontsize="30")

    # Plotting the Seed Points
    ax8 = fig.add_subplot(3, 3, 7)
    for seed in seeds:
        ax8.plot(seed[0], seed[1], marker='o', color="green", markersize=3)

    ax8.imshow(task2_gradient_image_copy, 'gray')
    ax8.set_title("Gradient Magnitude + Seed Points", fontsize="30")

    ax9 = fig.add_subplot(3, 3, 8)
    ax9.imshow(grown_img, 'gray')
    ax9.set_title("Grown img", fontsize="30")

    # Show Figure
    #fig.delaxes(axs[1][0])
    fig.tight_layout()
    fig.show()
    fig.savefig("output.pdf")


if __name__ == "__main__":
    main()
