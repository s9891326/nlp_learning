from sklearn import preprocessing
import scipy.sparse as sp

enc = preprocessing.OneHotEncoder()

# input
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
# print(enc.transform([[0, 1, 3]]).toarray())
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
print(X)

