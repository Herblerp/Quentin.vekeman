from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = None


def train_knn(neighbours, x_train, y_train, x_val, y_val):
    global knn
    knn = KNeighborsClassifier(n_neighbors=neighbours, n_jobs=-1)
    knn.fit(x_train, y_train)
    knn_accuracy = round(knn.score(x_val, y_val) * 100, 2)
    print("Accuracy for %d neighbours = %f" % (neighbours, knn_accuracy))


def predict_number(x_test, index):
    y_test_pred = knn.predict(x_test.values[index].reshape(1, -1))
    plt.imshow(x_test.values[index].reshape(28, 28))
    plt.show()
    print("The digit in the following image is ", y_test_pred[0])
