import data
import knn
import svm

data.shuffle_data()
data.deskew_data()

knn_iterations = 2
knn_accuracies = []

for x in range(knn_iterations):
    data.shuffle_data()
    knn_accuracies.append(knn.train_knn(3, data.x_train, data.y_train, data.x_val, data.y_val))

knn_average = sum(knn_accuracies) / len(knn_accuracies)
print("Average of knn for %d iterations is %f" % (knn_iterations, round(knn_average, 2)))


svm_iterations = 2
svm_accuracies = []

for x in range(svm_iterations):
    data.shuffle_data()
    svm_accuracies.append(svm.train_svm('rbf', 0, data.x_train, data.y_train, data.x_val, data.y_val))

svm_average = sum(svm_accuracies) / len(svm_accuracies)
print("Average of svm for %d iterations is %f" % (svm_iterations, round(svm_average, 2)))


