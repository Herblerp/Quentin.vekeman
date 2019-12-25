import data
import knn
import svm
import cnn

import plotly.graph_objs as go
import plotly.offline as plotly

data.load_data()
data.deskew_data()


def test_knn_neighbours(k_max, iterations, deskewed):
    print('*'.center(80, '*'))
    print(' KNN NEIGHBOURS TEST '.center(80))
    print('*'.center(80, '*'))
    print('Testing average accuracy for k in range [1, %d] with %d iterations each.' % (k_max, iterations))
    print('Deskewed: ' + str(deskewed))
    print()

    # Declare arrays for use in graph
    accuracies = []
    number_of_neighbors = []

    # Declare variables that store info about best amount of neighbours thus far
    best_accuracy = 0
    best_accuracy_k = 0

    for k in range(1, k_max + 1):

        # Array that keeps all accuracies for this value of k
        iteration_accuracies = []

        for iteration in range(iterations):

            print('-'.center(20, '-'))
            print('K = %d ITERATION %d'.center(20) % (k, iteration+1))
            print('-'.center(20, '-'))

            random_state = data.shuffle_data()
            print('Random_state = %d' % random_state)

            if deskewed:
                knn_accuracy = knn.train_knn(k, data.x_train_desk_shuffled, data.y_train_desk_shuffled,
                                             data.x_val_desk_shuffled, data.y_val_desk_shuffled)
            else:
                knn_accuracy = knn.train_knn(k, data.x_train_shuffled, data.y_train_shuffled, data.x_val_shuffled,
                                             data.y_val_shuffled)

            iteration_accuracies.append(round(knn_accuracy, 2))
            print('Accuracy = %f ' % knn_accuracy)
            print()

        # Calculate the average for current value of k
        average_accuracy = sum(iteration_accuracies) / len(iteration_accuracies)
        accuracies.append(average_accuracy)
        number_of_neighbors.append(k)

        print(' Average accuracy for K = %d is %f '.center(80, '*') % (k, average_accuracy))
        print()

        # Check if average is higher than current best
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_accuracy_k = k

    # Plot the results
    trace1 = go.Scatter(
        y=accuracies,
        x=number_of_neighbors,
        mode="lines",
        name="K-NN Classifier",
    )

    graph_data = [trace1]
    layout = dict(title='KNN Accuracy with descewed: ' + str(deskewed),
                  autosize=False,
                  width=800,
                  height=500,
                  yaxis=dict(title='Validation Accuracy (%)', gridwidth=2, gridcolor='#bdbdbd'),
                  xaxis=dict(title='Number of Neighbors', gridwidth=2, gridcolor='#bdbdbd'),
                  font=dict(size=14)
                  )
    fig = dict(data=graph_data, layout=layout)
    plotly.iplot(fig)

    # Print and return results
    print(' Best accuracy measured for K = %d with an accuracy of %f '.center(80, '*') % (
        best_accuracy_k, best_accuracy))
    print()

    return best_accuracy_k


# noinspection DuplicatedCode
def test_knn(k, iterations, deskewed):
    print('*'.center(80, '*'))
    print(' KNN ACCURACY TEST '.center(80))
    print('*'.center(80, '*'))
    print('Testing average accuracy for k=%d over %d iterations.' % (k, iterations))
    print('Deskewed: ' + str(deskewed))
    print()

    knn_accuracies = []

    for iteration in range(iterations):

        print('-'.center(20, '-'))
        print('ITERATION %d'.center(20) % (iteration+1))
        print('-'.center(20, '-'))

        random_state = data.shuffle_data()
        print('Random_state = %d' % random_state)

        if deskewed:
            accuracy = knn.train_knn(k, data.x_train_desk_shuffled, data.y_train_desk_shuffled,
                                     data.x_val_desk_shuffled, data.y_val_desk_shuffled)
        else:
            accuracy = knn.train_knn(k, data.x_train_shuffled, data.y_train_shuffled, data.x_val_shuffled,
                                     data.y_val_shuffled)

        knn_accuracies.append(accuracy)
        print('Accuracy = %f' % accuracy)
        print()

    knn_average = sum(knn_accuracies) / len(knn_accuracies)

    # Print and return results
    print(" Average accuracy for %d iterations is %f ".center(80, '*') % (iterations, round(knn_average, 2)))
    print()

    return knn_average


# Kernel possibilities: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
def test_svm(kernel, iterations, deskewed, grade=0):
    print('*'.center(80, '*'))
    print(' SVM ACCURACY TEST '.center(80))
    print('*'.center(80, '*'))
    if grade == 0:
        print('Testing average accuracy for %s kernel over %d iterations.' % (kernel, iterations))
    else:
        print('Testing average accuracy for %s kernel with grade %d over %d iterations.' % (kernel, grade, iterations))
    print('Deskewed: ' + str(deskewed))
    print()

    svm_accuracies = []

    for iteration in range(iterations):

        print('-'.center(20, '-'))
        print('ITERATION %d'.center(20) % (iteration+1))
        print('-'.center(20, '-'))

        random_state = data.shuffle_data()
        print('Random_state = %d' % random_state)

        if deskewed:
            accuracy = svm.train_svm(kernel,
                                     grade,
                                     data.x_train_desk_shuffled,
                                     data.y_train_desk_shuffled,
                                     data.x_val_desk_shuffled,
                                     data.y_val_desk_shuffled)
        else:
            accuracy = svm.train_svm(kernel,
                                     grade,
                                     data.x_train_shuffled,
                                     data.y_train_shuffled,
                                     data.x_val_shuffled,
                                     data.y_val_shuffled)

        svm_accuracies.append(accuracy)
        print('Accuracy = %f' % accuracy)

    svm_average = sum(svm_accuracies) / len(svm_accuracies)

    print(" Average of SVM with %s kernel for %d iterations is %f ".center(80, '*') % (
        kernel, iterations, round(svm_average, 2)))

    return svm_average


def test_nn(hidden_units, iterations, deskewed):
    return None