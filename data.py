import pandas
import random as rnd
import numpy as np
from random import seed
from random import randint
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation
from sklearn.model_selection import train_test_split

# Read the data files
data_train = pandas.read_csv("input/train.csv")
data_test = pandas.read_csv("input/test.csv")

# Modify the raw data
y_train = data_train.label.to_numpy()  # Convert the label column to a NumPy array.
x_train = data_train.drop(["label"], axis='columns')  # Drop the label column.

# Normalize the data
x_train = x_train / 255
x_test = data_test / 255

x_val = None
y_val = None

x_train_desk = None
y_train_desk = None
x_val_desk = None
y_val_desk = None
x_test_desk = None


def shuffle_data():
    
    rand = randint(0, 5000)
    print('Random_state set to %d' % rand)

    global x_train, x_val, y_train, y_val
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=rand)
    print("Dimention of x_train is " + str(x_train.shape))

    global x_train_desk, x_val_desk, y_train_desk, y_val_desk
    x_train_desk, x_val_desk, y_train_desk, y_val_desk = train_test_split(x_train_desk,
                                                                          y_train_desk,
                                                                          test_size=0.2,
                                                                          random_state=rand)


def print_raw_data_info():
    print(" RAW DATA INFO ".center(80, "#"))
    print(" TRAIN ".center(40, "_"))
    print("Shape: ", data_train.shape)
    print(data_train.head(5))  # Return the first n rows from the DataFrame.
    print(" TEST ".center(40, "_"))
    print("Shape: ", data_test.shape)
    print(data_test.head(5))


def deskew_data():

    global x_train_desk, x_test_desk, y_train_desk
    x_train_desk = pandas.DataFrame().reindex_like(x_train)
    x_test_desk = pandas.DataFrame().reindex_like(x_test)
    y_train_desk = np.copy(y_train)

    for xi in range(x_train.shape[0]):
        x_train_desk.values[xi] = deskew(x_train.values[xi].reshape(28, 28)).reshape(784)
    for xi in range(x_test.shape[0]):
        x_test_desk.values[xi] = deskew(x_test.values[xi].reshape(28, 28)).reshape(784)


def print_modified_data_info():
    print(" MODIFIED DATA INFO ".center(80, "#"))
    print(" Y-TRAIN ".center(40, "_"))
    print("Shape: ", y_train.shape)
    print(y_train)  # Array met labels
    print(" X-TRAIN ".center(40, "_"))
    print("Shape: ", x_train.shape)
    print(x_train)  # Train data without label column.
    print(" Y-VAL ".center(40, "_"))
    print("Shape: ", y_val.shape)
    print(y_val)  # Array met labels
    print(" X-VAL ".center(40, "_"))
    print("Shape: ", x_val.shape)
    print(x_val)  # Array met labels


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
