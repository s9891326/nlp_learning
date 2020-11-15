import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use Seaborn plotting defaults
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
# plt.plot([0.6], [2.1], "x", color="red", markeredgewidth=2, markersize=10)
#
# # 這3條非常不一樣的分割器，都可以完美的區分出這些樣本點，這樣要選哪一個呢?
# # 看一下"x"，會依據不同的分割器，被分類到不同類別
# # 所以在區間畫一條線是不夠的，需要在更加深入
# for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
#     plt.plot(xfit, m * xfit + b, "-k")
# plt.xlim(-1, 3.5)
# plt.show()

"""
svm 最大化邊界
與其簡單的在類別間劃一條寬度為0的線，我們可以在每一條線上畫上具有寬度的邊界
"""
xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
#
# for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
#     yfit = m * xfit + b
#     plt.plot(xfit, yfit, "-k")
#     plt.fill_between(xfit, yfit - d, yfit + d, edgecolor="none",
#                      color="#AAAAAA", alpha=0.4)
# plt.xlim(-1, 3.5)
# plt.show()

"""fit svm"""
from sklearn.svm import SVC
# model = SVC(kernel="linear", C=1E10)
# model.fit(X, y)

def plot_svc_decision_func(model, ax=None, plot_support=True):
    """plot the decision function for a two-dimensional SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # build grid to eval model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # draw decision borders
    ax.contour(X, Y, P, colors="k",
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=["--", "-", "--"])

    # draw svm
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors="none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
# plot_svc_decision_func(model)
# plt.show()
# print(model.support_vectors_)

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.6)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel="linear", C=1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_func(model, ax)

# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

"""SVM加入金的訓練資料點之影響"""
# 訓練模型
# for axi, N in zip(ax, [60, 120]):
#     plot_svm(N, axi)
#     axi.set_title(f"N = {N}")
# plt.show()


"""互動式SVM視覺圖的第一個畫面 - 沒結果"""
from ipywidgets import interact, fixed
# interact(plot_svm, N=[10, 200], ax=fixed(None))
# plt.show()

"""很難使用線性分類器的非線性邊界資料列"""
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
# clf = SVC(kernel="linear").fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
# plot_svc_decision_func(clf, plot_support=False)
# plt.show()

"""透過半徑基函數(RBF, radial basis function), 把資料投影到更高的維度"""
r = np.exp(-(X ** 2).sum(1))

from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap="autumn")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")

# interact(plot_3D(), elev=[-90, 90], azim=[-180, 180], X=fixed(X), y=fixed(y))
# plt.show()

"""核SVM擬合到資料"""
# clf = SVC(kernel="rbf", C=1E8)
# clf.fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
# plot_svc_decision_func(clf)
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=300, lw=1, facecolors="none")
# plt.show()

"""有某些程度上重疊的資料"""
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
# plt.show()

"""
C參數在SVM擬合中的影響
C參數最佳的值由資料集來決定，應該透過交叉驗證或類似的程序來調整
"""
flg, ax = plt.subplots(1, 2, figsize=(16, 6))
flg.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel="linear", C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
    plot_svc_decision_func(model=model, ax=axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors="none")
    axi.set_title(f"C = {C}", size=14)
plt.show()
