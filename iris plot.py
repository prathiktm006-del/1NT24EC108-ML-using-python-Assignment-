
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix

iris_dataset = load_iris()

print("Keys of iris dataset:\n{}".format(iris_dataset.keys()))


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)


scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3
)


knn = KNeighborsClassifier(n_neighbors=1)


knn.fit(X_train, y_train)


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new shape:", X_new.shape)


prediction = knn.predict(X_new)

print("Prediction:", prediction)
print("Predicted target name:", iris_dataset['target_names'][prediction])


y_pred = knn.predict(X_test)
print("Test set accuracy: {:.2f}".format(np.mean(y_pred == y_test)))
