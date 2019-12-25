import pandas
import random as rnd
import numpy as np
from random import seed
from random import randint
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation
from sklearn.model_selection import train_test_split

# Original data
x_train = None
y_train = None
x_test = None

# Deskewed data
x_train_desk = None
y_train_desk = None
x_test_desk = None

# Shuffled data
x_train_shuffled = None
y_train_shuffled = None
x_val_shuffled = None
y_val_shuffled = None

# Shuffled deskewed data
x_train_desk_shuffled = None
y_train_desk_shuffled = None
x_val_desk_shuffled = None
y_val_desk_shuffled = None


# Load and prepare the raw data from the csv files
def load_data():
    global x_train, y_train, x_test

    # Read the data files
    data_train = pandas.read_csv("input/train.csv")
    data_test = pandas.read_csv("input/test.csv")

    # Modify the raw data
    y_train = data_train.label.to_numpy()  # Convert the label column to a NumPy array.
    x_train = data_train.drop(["label"], axis='columns')  # Drop the label column.

    # Normalize the data
    x_train = x_train / 255
    x_test = data_test / 255


# Deskew the images from the original data
def deskew_data():
    global y_train, x_train_desk, y_train_desk, x_test_desk

    x_train_desk = pandas.DataFrame().reindex_like(x_train)
    x_test_desk = pandas.DataFrame().reindex_like(x_test)
    y_train_desk = np.copy(y_train)

    for xi in range(x_train.shape[0]):
        x_train_desk.values[xi] = deskew(x_train.values[xi].reshape(28, 28)).reshape(784)
    for xi in range(x_test.shape[0]):
        x_test_desk.values[xi] = deskew(x_test.values[xi].reshape(28, 28)).reshape(784)


# Split and shuffle the normal and deskewed data using the same random seed (so same-indexed images are equal)
def shuffle_data():
    rand = randint(0, 5000)
    print('Random_state set to %d' % rand)

    global x_train, y_train, x_train_shuffled, y_train_shuffled, x_val_shuffled, y_val_shuffled

    x_train_shuffled, x_val_shuffled, y_train_shuffled, y_val_shuffled = train_test_split(x_train,
                                                                                          y_train,
                                                                                          test_size=0.2,
                                                                                          random_state=rand)

    global x_train_desk, y_train_desk, x_train_desk_shuffled, x_val_desk_shuffled, y_train_desk_shuffled, \
        y_val_desk_shuffled

    x_train_desk_shuffled, x_val_desk_shuffled, y_train_desk_shuffled, y_val_desk_shuffled = train_test_split(
        x_train_desk,
        y_train_desk,
        test_size=0.2,
        random_state=rand)


def print_data_info():
    print(" DATA INFO ".center(80, "#"))
    print(" Y-TRAIN ".center(40, "_"))
    print("Shape: ", y_train.shape)
    print(y_train)  # Array met labels
    print(" X-TRAIN ".center(40, "_"))
    print("Shape: ", x_train.shape)
    print(x_train)  # Train data without label column.


def show_sample_image(x_data, index):
    image1 = x_data.values[index].reshape(28, 28)  # Reshape 1*784 data to 28*28 image
    plt.imshow(image1)
    plt.show()


def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    img = interpolation.affine_transform(image, affine, offset=offset)
    return (img - img.min()) / (img.max() - img.min())


def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]  # A trick in numPy to create a mesh grid
    total_image = np.sum(image)  # sum of pixels
    m0 = np.sum(c0 * image) / total_image  # mu_x
    m1 = np.sum(c1 * image) / total_image  # mu_y
    m00 = np.sum((c0 - m0) ** 2 * image) / total_image  # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / total_image  # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / total_image  # covariance(x,y)
    mu_vector = np.array([m0, m1])  # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00, m01], [m01, m11]])  # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix
