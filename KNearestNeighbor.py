from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading data and splitting features and labels
iris = datasets.load_iris()
features = iris.data
labels = iris.target

# print(features[0],"Label: ",labels[0])
# print(iris.DESCR)

# Training classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[31, 1, 1, 1]])

print(preds)