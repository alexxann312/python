import numpy as np

from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# labeltext = ["apple", "apple", "orange", "orange"]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[160,0]])

### iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
test_idx = [0, 50, 100]

print iris.feature_names
print iris.target_names

print iris.data[0]
print iris.target[0]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print test_target
print test_data
print clf.predict(test_data)

#visualising a decision tree

import pydotplus
from IPython.display import Image
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")


