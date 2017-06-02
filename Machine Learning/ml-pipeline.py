# import packages
import numpy as np

# import dataset
### iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

## Decision Tree Classifier
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#import random
from scipy.spatial import distance
from scipy import stats
from array import array

def euc(a, b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            #label = random.choice(self.y_train)
            label = self.k_closest(row, 5) # change the number here to increase the number of neighbours
            predictions.append(label)
        return np.asarray(predictions)

    def k_closest(self, row, k):
        knn_label = []
        neighbor_dist = []
        for i in self.X_train:
            dist = euc(row, i)
            neighbor_dist.append(dist)  # compute the distance from all neighbors

        ndist = np.array(neighbor_dist)
        knn = ndist.argsort()[:k]  # find the index of the 3 closest values

        for j in knn:
            knn_label.append(self.y_train[j])  # categorising

        pred = stats.mode(knn_label)[0][0]  # finding the most frequently occured values
        return pred.astype(int) #convert the value back to int as mode return float values


    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]




## KNearestNeighbours
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print predictions
print y_test

print type(predictions), type(y_train)


from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)


# print len(y_train)
#
# for row in range(1, len(y_train)):
#
#     knn = []
#     knn_label = []
#     neighbor_dist = []
#     for i in range(0, len(X_train)):
#         dist = euc(row, X_train[i])
#         neighbor_dist.append(dist)  #compute the distance from all neighbors
#
#     ndist = np.array(neighbor_dist)
#
#     knn = np.argpartition(ndist, -3)[-3:] #find the index of the 3 closest values
#
#     for j in knn:
#         knn_label.append(y_train[j]) #categorising
#
#     pred = stats.mode(knn_label)[0][0] #finding the most frequently occured values
#
#     print pred