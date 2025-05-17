# Training a logistic regressor to identify whether a flower is iris virginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

# print(iris.keys())
# print(iris.DESCR)
# print(iris.data)
# print(iris.target)
# print(iris.data.shape)

X = iris.data[:, 3:]
Y = (iris.target==2).astype(int) #saving as 0 or 1 value

# training logistic regression classifier
clf = LogisticRegression()
clf.fit(X, Y)

example = clf.predict(( [ [1.6] ] ))

print(example)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
Y_prob = clf.predict_proba(X_new)
plt.plot(X_new, Y_prob[:, 1])
plt.show()