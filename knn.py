from sklearn.neighbors import KNeighborsClassifier


def train_knn(neighbours, x_train, y_train, x_val, y_val):
    knn = KNeighborsClassifier(n_neighbors=neighbours, n_jobs=-1)
    knn.fit(x_train, y_train)
    knn_accuracy = round(knn.score(x_val, y_val)*100, 2)
    print("Accuracy for %d neighbours = %f" % (neighbours, knn_accuracy))
