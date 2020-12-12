import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

"""decision tree"""
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
# plt.show()

""""""
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)

def visulize_classifier(model, X, y, ax=None, cmap="rainbow"):
    ax = ax or plt.gca()

    # 繪出訓練用資料點
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap)
    ax.axis("tight")
    ax.axis("off")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 擬合一個評估器
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 建立結果的彩色圖形
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, ylim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)

# visulize_classifier(DecisionTreeClassifier(), X, y)
# plt.show()

"""進行邊界決策的一個整體隨機決策樹"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
                        random_state=1)
# bag.fit(X, y)
# visulize_classifier(bag, X, y)
# plt.show()

"""以random forest進行邊界決策"""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
# visulize_classifier(model, X, y)
# plt.show()

"""隨機森林迴歸"""
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)

def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))

    return slow_oscillation + fast_oscillation + noise
# y = model(x)
# plt.errorbar(x, y, 0.3, fmt="o")
# plt.show()

"""使用隨機森林迴歸"""
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
# forest.fit(x[:, None], y)
#
# xfit = np.linspace(0, 10, 1000)
# yfit = forest.predict(xfit[:, None])
# yture = model(xfit, sigma=0)
#
# plt.errorbar(x, y, 0.3, fmt="o", alpha=0.5)
# plt.plot(xfit, yfit, "-r")
# plt.plot(xfit, yture, "-k", alpha=0.5)
# plt.show()

"""範例: 隨機森林來做數字元分類"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

digits = load_digits()
print(digits.keys())

# 設定fig
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation="nearest")
#
#     # 對目標值標上標籤
#     ax.text(0, 7, str(digits.target[i]))
# plt.show()

Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print(classification_report(y_pred=ypred, y_true=ytest))

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("pred label")
plt.show()
