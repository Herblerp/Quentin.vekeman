import data
import svm
data.deskew_data()

# Similar results with svm rbf kernel
svm.train_svm('rbf', 4, data.x_train, data.y_train, data.x_val, data.y_val)

# Some more test data predictions
# svm.predict_number(data.x_test, 0)
# svm.predict_number(data.x_test, 2)