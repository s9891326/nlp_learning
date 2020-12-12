import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

sns.set()


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
# print(X.shape)

# plt.scatter(X[:, 0], X[:, 1])
# plt.axis("equal")
# plt.show()

# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.components_)
# print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle="->",
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate("", v1, v0, arrowprops=arrowprops)

# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     print(f"v: {v}")
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis("equal")
# plt.show()

"""降維"""
# pca = PCA(n_components=1)
# pca.fit(X)
# x_pca = pca.transform(X)
# print(f"original shape: {X.shape}")
# print(f"transformed shape: {x_pca.shape}")

"""對降維後的資料進行反轉"""
# x_new = pca.inverse_transform(x_pca)
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# plt.scatter(x_new[:, 0], x_new[:, 1], alpha=0.8)
# plt.axis("equal")
# plt.show()

"""PCA 手寫數字"""
from sklearn.datasets import load_digits
digits = load_digits()

# pca = PCA(n_components=2)
# projected = pca.fit_transform(digits.data)
# print(digits.data.shape)
# print(projected.shape)
#
# plt.scatter(projected[:, 0], projected[:, 1],
#             c=digits.target, edgecolors="none", alpha=0.5,
#             cmap=plt.cm.get_cmap("Spectral", 10))
# plt.xlabel("component1")
# plt.ylabel("component2")
# plt.colorbar()
# plt.show()

"""PCA做雜訊過濾"""
def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap="binary", interpolation="nearest",
                  clim=(0, 16))

# plot_digits(digits.data)
# plt.show()

# 加上高斯隨機雜訊的數字元
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
# plot_digits(noisy)
# plt.show()

# pca保留50% 變異量
pca = PCA(0.5).fit(noisy)
# print(pca.n_components_)  # 50%的變異量需要12個主要成分

# 計算成分取反轉此轉換以重建過濾過的數字元
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
# plot_digits(filtered)
# plt.show()

"""特徵臉"""
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

pca = PCA(150)
pca.fit(faces.data)
# fig, axes = plt.subplots(3, 8, figsize=(9, 4),
#                          subplot_kw={"xticks": [], "yticks": []},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(62, 47), cmap="bone")
# plt.show()

# LFW 資料的累積已解釋變異量
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("number of components")
# plt.ylabel("cumulative explained variance")
# plt.show()

# 計算成分和被投射的臉
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)
fig, ax = plt.subplots(2, 10, figsize=(9, 3),
                       subplot_kw={"xticks": [], "yticks": []},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap="binary_r")
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap="binary_r")

ax[0, 0].set_ylabel("full-dim\ninput")
ax[1, 0].set_ylabel("150-dim\nreconstruction")
plt.show()
