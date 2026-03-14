import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
cancer = load_breast_cancer()

# Select only first 2 features (required for 2D plotting)
X = cancer.data[:, :2]
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create Random Forest model
forest = RandomForestClassifier(n_estimators=5, random_state=2)

# Train model
forest.fit(X_train, y_train)

# Create plot area
fig, axes = plt.subplots(2, 3, figsize=(20,10))

# Plot each tree
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

# Plot Random Forest decision boundary
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1,-1], alpha=.4)
axes[-1,-1].set_title("Random Forest")

plt.show()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Random Forest Model
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# Function to plot feature importance
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.figure(figsize=(8,10))
    plt.barh(range(n_features), model.feature_importances_)
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Breast Cancer Dataset")
    plt.show()

# Call the function
plot_feature_importances_cancer(forest)
