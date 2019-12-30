import data
import knn
import svm
import cnn

import plotly.graph_objs as go
import plotly.offline as py

py.init_notebook_mode(connected=True)

data.load_data()
data.deskew_data()
data.shuffle_data()


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
            print('K = %d ITERATION %d'.center(20) % (k, iteration + 1))
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

        print((' Average accuracy for K = %d is %f ' % (k, average_accuracy)).center(80, '#'))
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
    layout = dict(title='KNN average accuracy for amount of neighbours | Descewed: ' + str(deskewed),
                  autosize=False,
                  width=800,
                  height=500,
                  yaxis=dict(title='Validation Accuracy (%)', gridwidth=2, gridcolor='#bdbdbd'),
                  xaxis=dict(title='Number of Neighbors', gridwidth=2, gridcolor='#bdbdbd'),
                  font=dict(size=14)
                  )
    fig = dict(data=graph_data, layout=layout)
    py.iplot(fig)

    # Print and return results
    print((' Best accuracy measured for K = %d with an accuracy of %f ' % (
        best_accuracy_k, best_accuracy)).center(80, '#'))
    print()
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
        print('ITERATION %d'.center(20) % (iteration + 1))
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
    print((' Average accuracy for %d iterations is %f ' % (iterations, round(knn_average, 2))).center(80, '#'))
    print()
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
        print('ITERATION %d'.center(20) % (iteration + 1))
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

    print((' Average of SVM with %s kernel for %d iterations is %f ' % (
        kernel, iterations, round(svm_average, 2))).center(80, '#'))

    return svm_average


def test_nn(layers_min, layers_max, layers_interval, hu_min, hu_max, hu_interval, batch_size, optimizer, epochs,
            deskewed):
    print('*'.center(80, '*'))
    print(' NN TEST '.center(80))
    print('*'.center(80, '*'))
    print('Testing accuracy for neural net with %s optimizer for %d epochs' % (optimizer, epochs))
    print('Deskewed: ' + str(deskewed))
    print()

    iteration = 1
    max_acc = 0
    max_acc_layers = 0
    max_acc_hu = 0

    rand = data.shuffle_data()
    print('Random_state: %d' % rand)

    for layers in range(layers_min, layers_max + 1, layers_interval):
        for hu in range(hu_min, hu_max + 1, hu_interval):
            print('-'.center(20, '-'))
            print('ITERATION %d'.center(20) % iteration)
            print('-'.center(20, '-'))
            print('Batch_size = %d' % batch_size)
            print('Hidden_layers = %d' % layers)
            print('Hidden_units = %d' % hu)
            if deskewed:
                accuracy = cnn.train_nn(data.x_train_desk_shuffled,
                                        data.y_train_desk_shuffled,
                                        data.x_val_desk_shuffled,
                                        data.y_val_desk_shuffled,
                                        batch_size, epochs, hu, layers, optimizer, 0)
            else:
                accuracy = cnn.train_nn(data.x_train_shuffled,
                                        data.y_train_shuffled,
                                        data.x_val_shuffled,
                                        data.y_val_shuffled,
                                        batch_size, epochs, hu, layers, optimizer, 0)
            print('Accuracy = %f' % round(accuracy, 2))
            iteration += 1
            if accuracy > max_acc:
                max_acc = accuracy
                max_acc_hu = hu
                max_acc_layers = layers

    print('-'.center(20, '-'))
    print(' BEST RESULT '.center(80, '#'))
    print('-'.center(20, '-'))
    print('Accuracy = %f' % max_acc)
    print('Hidden_layers = %d' % max_acc_layers)
    print('Hidden_units = %d' % max_acc_hu)

    return max_acc


def test_cnn(deskewed):
    print('*'.center(80, '*'))
    print(' NN TEST '.center(80))
    print('*'.center(80, '*'))
    print('Testing accuracy for CNN over 30 epochs')

    rand = data.shuffle_data()
    print('Random_state: %d' % rand)
    print('Deskewed: ', str(deskewed))

    if deskewed:
        cnn.train_cnn(data.x_train_desk_shuffled, data.y_train_desk_shuffled, data.x_val_desk_shuffled,
                      data.y_val_desk_shuffled, 86, 30)
    else:
        cnn.train_cnn(data.x_train_shuffled, data.y_train_shuffled, data.x_val_shuffled, data.y_val_shuffled, 86, 30)
