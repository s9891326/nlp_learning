from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


digits = load_digits()
print(digits.images.shape)


"""show picture"""
# fig, axes = plt.subplots(10, 10, figsize=(8, 8),
#                          subplot_kw={"xticks": [], "yticks": []},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
#
# for i, ax in enumerate(axes.flat):
#     ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
#     ax.text(0.05, 0.05, str(digits.target[i]),
#             transform=ax.transAxes, color="green")
# plt.show()


"""preprocess data"""
# X = digits.data
# print(X.shape)

# y = digits.target
# print(y.shape)

"""dimensionality reduction"""
# iso = Isomap(n_components=2)
# iso.fit(X)
# data_projected = iso.transform(X)
# data_projected = iso.fit_transform(X)
# print(data_projected.shape)

"""show dimensionality reduction picture"""
# plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
#             edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Spectral', 10))
# plt.colorbar(label="digit label", ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()

"""classification"""
# xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
# model = GaussianNB()
# model.fit(xtrain, ytrain)
# y_model = model.predict(xtest)
# accuracy = accuracy_score(ytest, y_model)
# print(accuracy)

"""show confusion matrix"""
# mat = confusion_matrix(ytest, y_model)
# sns.heatmap(mat, square=True, annot=True, cbar=False)
# plt.xlabel("predicted value")
# plt.ylabel("true value")
# plt.show()

"""cross_val_score"""
# model = GaussianNB()
# score = cross_val_score(model, X, y, cv=5)
# print(score)

"""leave-one-out cross validation"""
# print(len(X))
# model = GaussianNB()
# scores = cross_val_score(model, X, y, cv=LeaveOneOut())
# print(scores)
# print(scores.mean())


"""validation line"""
# y = ax + b (degree = 1)
# y = ax^3 + bx^2 + cx + d (degree = 3)

def polynomial_regression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    # print(f"X={X}")
    y = 10 - 1. / (X.ravel() + 0.1)
    # print(f"y={y}")
    if err > 0:
        y += err * rng.randn(N)
    return X, y

"""show model fit"""
# X, y = make_data(40)
# x_test = np.linspace(-0.1, 1.1, 500)[:, None]

# plt.scatter(X.ravel(), y, color="black")
# axis = plt.axis()
# for degree in [1, 3, 5]:
#    y_test = polynomial_regression(degree).fit(X, y).predict(x_test)
#    plt.plot(x_test.ravel(), y_test, label=f'degree={degree}')
# plt.xlim(-0.1, 1.0)
# plt.ylim(-2, 12)
# plt.legend(loc="best")
# plt.show()

"""validation curve"""
# degree = np.arange(0, 21)
# train_score, val_score = validation_curve(polynomial_regression(), X, y,
#                                           "polynomialfeatures__degree",
#                                           degree, cv=7)
# plt.plot(degree, np.median(train_score, 1), color="blue", label="training score")
# plt.plot(degree, np.median(val_score, 1), color="red", label="validation score")
# plt.legend(loc="best")
# plt.ylim(0, 1)
# plt.xlabel("degree")
# plt.ylabel("score")
# plt.show()

# plt.scatter(X.ravel(), y)
# lim = plt.axis()
# y_test = polynomial_regression(3).fit(X, y).predict(x_test)
# plt.plot(x_test.ravel(), y_test)
# plt.axis(lim)
# plt.show()


"""learning curve"""
# X2, Y2 = make_data(200)
# # plt.scatter(X2.ravel(), Y2)
# # plt.show()
# degree = np.arange(21)
# train_score2, val_score2 = validation_curve(polynomial_regression(), X2, Y2,
#                                           "polynomialfeatures__degree",
#                                           degree, cv=7)
# plt.plot(degree, np.median(train_score2, 1), color="blue", label="training score")
# plt.plot(degree, np.median(val_score2, 1), color="red", label="validation score")
# plt.plot(degree, np.median(train_score, 1), color="blue", alpha=0.3, linestyle="dashed")
# plt.plot(degree, np.median(val_score, 1), color="red", alpha=0.3, linestyle="dashed")
# plt.legend(loc="best")
# plt.ylim(0, 1)
# plt.xlabel("degree")
# plt.ylabel("score")
# plt.show()


"""feature engineering"""
# from sklearn.feature_extraction import DictVectorizer
# data = [
#     {"price": 850000, "rooms": 4, "neighborhood": "Queen Anne"},
#     {"price": 700000, "rooms": 3, "neighborhood": "Fremont"},
#     {"price": 650000, "rooms": 3, "neighborhood": "Wallingford"},
#     {"price": 600000, "rooms": 2, "neighborhood": "Fremont"},
# ]
#
# vec = DictVectorizer(sparse=False, dtype=int)
# vec.fit_transform(data)
# feature_names = vec.get_feature_names()
#
# print(vec.fit_transform(data))
# # print(sparse_matrix)

"""text features --> TF-IDF(term frequency-inverse document frequency) """
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import pandas as pd
#
# sample = ["problem of evil", "evil queen", "horizon problem"]
# vec = CountVectorizer()
# X = vec.fit_transform(sample)
# print(X)
#
# df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
# print(df)
#
# tfidf_vec = TfidfVectorizer()
# X2 = tfidf_vec.fit_transform(sample)
# df2 = pd.DataFrame(X2.toarray(), columns=tfidf_vec.get_feature_names())
# print(df2)

"""image features"""
# identity face pipeline

"""derived features ---> kernel methods --> SVM"""
# 透過轉換輸入而不是改變模型來把線性迴歸轉變成多項式迴歸
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([4, 2, 1, 3, 7])
# plt.scatter(x, y)
# # plt.show()
#
# X = x[:, np.newaxis]
# model = LinearRegression().fit(X, y)
# yfit = model.predict(X)
# plt.plot(x, yfit)
# # plt.show()
#
# poly = PolynomialFeatures(degree=3, include_bias=False)
# X2 = poly.fit_transform(X)
# model = LinearRegression().fit(X2, y)
# yfit = model.predict(X2)
# plt.plot(x, yfit, color="r")
# plt.show()


"""imputation missing data"""
# 藉由imputer 解決缺失的資料(nan)
# x = np.array([[np.nan, 0, 3],
#               [3, 7, 9],
#               [3, 5, 2],
#               [4, np.nan, 6],
#               [8, 8, 1]])
# y = np.array([14, 16, -1, 8, -5])
#
# imp = SimpleImputer(strategy="mean")
# print(f"x: {x}")
# x2 = imp.fit_transform(x)
# print(f"x2: {x2}")
#
# model = LinearRegression().fit(x2, y)
# y2 = model.predict(x2)
# print(y2)

"""pipeline features"""
# 1. 使用平均值替補缺失資料
# 2. 轉換特徵成為二階方程式
# 3. 擬和線性迴歸
# 將自動套用，形成並行的管線
# model = make_pipeline(Imputer(strategy="mean"),
#                       PolynomialFeatures(degree=2),
#                       LinearRegression())
# model.fit(x, y)
# print(y)
# print(model.predict(x))
