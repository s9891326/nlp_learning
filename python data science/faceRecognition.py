from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

faces = fetch_lfw_people(min_faces_per_person=60)
# print(faces)
print(faces.target_names)
print(faces.images.shape)

# fig, ax = plt.subplots(3, 5)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap="bone")
#     axi.set(xticks=[], yticks=[],
#             xlabel=faces.target_names[faces.target[i]])
# plt.show()

pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel="rbf", class_weight="balanced")
model = make_pipeline(pca, svc)

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

# C:用來控制邊界的銳利度，gamma:用來控制RBF kernel的大小
param_grid = {"svc__C": [1, 5, 10, 50],
              "svc__gamma": [0.0001, 0.005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

grid.fit(Xtrain, ytrain)
print(f"best param: {grid.best_params_}")

model = grid.best_estimator_
y_pred = model.predict(Xtest)

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap="bone")
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[y_pred[i]],
                   color="black" if y_pred[i] == ytest[i] else "red")
fig.suptitle("predicted names: incorrect labels in red", size=14)
plt.show()

print(classification_report(ytest, y_pred,
                            target_names=faces.target_names))

mat = confusion_matrix(ytest, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel("true label")
plt.ylabel("pred label")
plt.show()
