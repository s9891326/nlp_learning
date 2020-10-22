"""樸素貝氏分類法"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

categories = ["talk.religion.misc", "soc.religion.christian", "sci.space", "comp.graphics"]


class NaiveBayesClassification:
    def __init__(self):
        self.train = fetch_20newsgroups(subset="train", categories=categories)
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    def fit(self):
        self.model.fit(self.train.data, self.train.target)

    def predict(self, target, data):
        labels = self.model.predict(data)
        self.show_heatmap(target, labels)

    def predict_category(self, s):
        pred = self.model.predict([s])
        print(pred[0])
        return self.train.target_names[pred[0]]

    def show_heatmap(self, target, labels):
        mat = confusion_matrix(target, labels)
        sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False,
                    xticklabels=self.train.target_names, yticklabels=self.train.target_names)
        plt.xlabel("true label")
        plt.ylabel("predicted label")
        plt.show()

if __name__ == '__main__':
    test = fetch_20newsgroups(subset="test", categories=categories)
    naiveBayes = NaiveBayesClassification()
    naiveBayes.fit()
    # naiveBayes.predict(test.target, test.data)

    print(naiveBayes.predict_category("sending a payload to the iss"))

