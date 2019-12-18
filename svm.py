from sklearn.svm import SVC
from matplotlib import pyplot as plt
svm = None


def train_svm(kernel, degree, x_train, y_train, x_val, y_val):
    global svm
    svm = SVC(kernel=kernel, random_state=1, degree=degree)
    svm.fit(x_train, y_train)
    SVM_accuracy = svm.score(x_val, y_val) * 100
    SVM_accuracy = round(SVM_accuracy, 2)

    print("SVM_accuracy is %", SVM_accuracy)

def predict_number(x_test, index):
    y_test_pred = svm.predict(x_test.values[index].reshape(1, -1))
    plt.imshow(x_test.values[index].reshape(28, 28))
    plt.show()
    print("The digit in the following image is ", y_test_pred[0])