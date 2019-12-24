import data
import knn
import svm
import cnn

import plotly.graph_objs as go
import plotly.offline as plotly

data.shuffle_data()
data.deskew_data()


def test_knn_neighbours(k_max, iterations):
    accuracies = []
    number_of_neighbors = []
    best_accuracy = 0
    best_accuracy_k = 0
    for k in range(1, k_max+1):
        iteration_accuracies = []
        for iteration in range(1, iterations+1):
            data.shuffle_data()
            knn_accuracy = knn.train_knn(k, data.x_train, data.y_train, data.x_val, data.y_val)
            iteration_accuracies.append(round(knn_accuracy, 2))
            print('Accuracy for iteration %d with %d neighbours is %f' % (iteration, k, knn_accuracy))

        average_accuracy = sum(iteration_accuracies) / len(iteration_accuracies)
        print('Average accuracy for %d neighbours is %f' % (k, average_accuracy))
        accuracies.append(average_accuracy)
        number_of_neighbors.append(k)

        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_accuracy_k = k

    trace1 = go.Scatter(
        y=accuracies,
        x=number_of_neighbors,
        mode="lines",
        name="K-NN Classifier",
    )

    graph_data = [trace1]
    layout = dict(title='KNN Accuracy',
                  autosize=False,
                  width=800,
                  height=500,
                  yaxis=dict(title='Validation Accuracy (%)', gridwidth=2, gridcolor='#bdbdbd'),
                  xaxis=dict(title='Number of Neighbors', gridwidth=2, gridcolor='#bdbdbd'),
                  font=dict(size=14)
                  )
    fig = dict(data=graph_data, layout=layout)
    plotly.iplot(fig)

    print('Best accuracy measured for %d neighbours with an accuracy of %f' % (best_accuracy_k, best_accuracy))
    return best_accuracy_k


def knntest():
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


