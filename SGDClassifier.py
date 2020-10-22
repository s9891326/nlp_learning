import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets

# iris = datasets.load_iris()
# x, y = iris.data, iris.target
# # print(f"x = {x},\n y = {y}")
#
# clf = SGDClassifier(alpha=0.001, max_iter=100).fit(x, y)
# y_pred = clf.predict(x)
# print(f"準確率: {accuracy_score(y, y_pred)}")
# print(f"二分類索引: {clf.classes_}")  # one versus all 方法來組合多個二分類器
# print(f"迴歸係數: {clf.coef_}")  # 每一個二分類器的迴歸係數
# print(f"偏差: {clf.intercept_}")  # 每一個二分類器的偏差
#
# print(x.shape)
# print(y.shape)

# we create 50 separable points
X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# fit the model
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)
clf.fit(X, Y)

# plot the line, the points, and the nearest vectors to the plane
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx, yy)
# print(f"x1 : {X1}, \n x2: {X2}")
Z = np.empty(X1.shape)

for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    print(f"x1: {x1}, x2: {x2}")
    p = clf.decision_function([[x1, x2]])
    print(p)
    Z[i, j] = p[0]
    print(i, j)
    if j == 1:
        print(Z)
        print(Z[0, 2])
        break

# levels = [-1.0, 0.0, 1.0]
# linestyles = ['dashed', 'solid', 'dashed']
# colors = "k"
# print(f"z: {Z}")
# plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="black", s=20)
# plt.axis("tight")
# plt.show()

# plt.scatter(tempx, tempy)
# plt.show()

