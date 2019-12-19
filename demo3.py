import data
import cnn

# Classic NN with 2 layers before deskewing
print("Classic NN with 2 layers before deskewing")
cnn.train_nn(data.x_train,
             data.y_train,
             data.x_val,
             data.y_val,
             batch_size=250,
             epochs=50,
             hidden_units=300,
             verbosity=0)

# Classic NN with 2 layers after deskewing
print("Classic NN with 2 layers after deskewing")
data.deskew_data()

cnn.train_nn(data.x_train,
             data.y_train,
             data.x_val,
             data.y_val,
             batch_size=250,
             epochs=50,
             hidden_units=300,
             verbosity=0)

# Complex CNN
print("Complex CNN")
cnn.train_cnn(data.x_train, data.y_train, data.x_val, data.y_val, batch_size=86, epochs=2)

# Next: Further tweak cnn / design my own and export the data
