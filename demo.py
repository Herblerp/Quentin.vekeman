import data
import knn

data.show_image_samples()
data.descew_data()
data.show_image_samples()

# # Prepare the data
# data.print_raw_data_info()
# data.print_modified_data_info()
#
# # Example before skewing
# data.show_image_samples()
#
# # # Train before skewing
# # knn.train_knn(3, data.x_train, data.y_train, data.x_val, data.y_val)
#
# # Predict a number from the test set
# # knn.predict_number(data.x_test, 0)
#
# # Deskew the data
#data.descew_data()
#
# # Same examples after deskewing
# data.show_image_samples()
# knn.train_knn(3, data.x_train, data.y_train, data.x_val, data.y_val)
#
# # Predict a number from the test set
# knn.predict_number(data.x_test, 0)
# knn.predict_number(data.x_test, 2)

# Knn_accuracies = []
# number_of_neighbors = []
# for neighbours in range(1, 10):
#     Knn = KNeighborsClassifier(n_neighbors=neighbours)
#     Knn.fit(data.x_train, data.y_train)
#     Knn_accuracy = round(Knn.score(data.x_val, data.y_val)*100,2)
#     Knn_accuracies.append(Knn_accuracy)
#     number_of_neighbors.append(neighbours)
#     print("Accuracy for %d neighbours = %f" % (neighbours, Knn_accuracy))
