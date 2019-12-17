import pandas
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Read the data files
data_train = pandas.read_csv("input/train.csv")
data_test = pandas.read_csv("input/test.csv")

# Check the first few lines of the files
print("*****RAW DATA*****")
print("__Train__")
print(data_train.head(5))  # Return the first n rows from the DataFrame.
print("__Test__")
print(data_test.head(5))

# Check the names of the columns
# print(data_train.columns) #The column labels of the DataFrame.
# print(data_test.columns)

# Modify the raw data
y_train = data_train.label.to_numpy()  # Convert the label column to a NumPy array.
x_train = data_train.drop(["label"], axis='columns')  # Drop the label column.

# Check the modified data
print("*****MODIFIED DATA*****")
print("__Labels__")
print(y_train)  # Array met labels
print("__Pixel Values__")
print(x_train)  # Train data zonder label kolom.

# Normalize the data
x_train = x_train / 255
x_test = data_test / 255

# Display a sample image
image1 = x_train.values[0].reshape(28, 28)  # Reshape 1*784 data to 28*28 image
image2 = x_train.values[3].reshape(28, 28)
plt.imshow(image1)
#plt.show()
plt.imshow(image2)
#plt.show()

# Check the shape of the test data
print("*****TEST DATA SHAPE*****")
print("x_test:", x_test.shape)  # Test data. These are the values we will have to label.

# Split the modified train set for validation.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

#Check the shape of the train and validation data
print("*****TRAIN DATA SHAPES*****")
print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("*****VALIDATION DATA SHAPES*****")
print("x_val: ",x_val.shape)
print("y_val: ",y_val.shape)