
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


X, y = mglearn.datasets.make_wave(n_samples=40)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


fig, axes = plt.subplots(1, 3, figsize=(15,5))

line = np.linspace(-3, 3, 1000).reshape(-1,1)

for n_neighbors, ax in zip([1,3,9], axes):
    
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)

    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', markersize=8)
    ax.plot(X_test, y_test, 'v', markersize=8)

    ax.set_title(
        "{} neighbors\nTrain score: {:.2f} Test score: {:.2f}".format(
            n_neighbors,
            reg.score(X_train, y_train),
            reg.score(X_test, y_test)
        )
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axes[0].legend(["Model predictions","Training data","Test data"], loc="best")

plt.show()
