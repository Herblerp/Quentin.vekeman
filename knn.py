from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

knn = None


def train_knn(neighbours, x_train, y_train):
    global knn
    knn = KNeighborsClassifier(n_neighbors=neighbours, n_jobs=-1)
    knn.fit(x_train, y_train)


def predict_number(x_test, index):
    y_test_pred = knn.predict(x_test.values[index].reshape(1, -1))
    plt.imshow(x_test.values[index].reshape(28, 28))
    plt.show()
    print("The digit in the following image is ", y_test_pred[0])


def save_knn(filename):
    dump(knn, filename)


def load_knn(filename):
    global knn
    knn = load(filename)


def calculate_accuracy(x_val, y_val):
    knn_accuracy = round(knn.score(x_val, y_val) * 100, 2)
    print("Accuracy = %f" % knn_accuracy)
