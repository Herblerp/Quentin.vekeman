import data
import knn

# Deskew the data
data.deskew_data()

# KNN after deskewing
knn.train_knn(3, data.x_train, data.y_train, data.x_val, data.y_val)

# Some test data predictions
# data.show_image_samples()
# knn.predict_number(data.x_test, 0)
# knn.predict_number(data.x_test, 2)