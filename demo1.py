import data
import knn
import svm


#CONSOLE1

# KAGGLE DIGIT RECOGNIZER CHALLENGE
# Label handwritten digits correctly

# Raw data (in csv format)
data.print_raw_data_info()
data.show_image_samples()

# Normalize and split the data
data.print_modified_data_info()

# KNN before deskewing
knn.train_knn(3, data.x_train, data.y_train, data.x_val, data.y_val)

# Some test data predictions
knn.predict_number(data.x_test, 0)
knn.predict_number(data.x_test, 2)

# Deskew the data
data.deskew_data()
data.show_image_samples()

# KNN after deskewing
knn.train_knn(3, data.x_train, data.y_train, data.x_val, data.y_val)

# Some test data predictions
knn.predict_number(data.x_test, 0)
knn.predict_number(data.x_test, 2)

