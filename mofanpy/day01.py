import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import itertools

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(",", "").split(" ") for d in docs]
vocab = set(itertools.chain(*docs_words))
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {v: i for i, v in v2i.items()}
print(f"docs_words{docs_words}")
print(f"v2i:{v2i}")
print(f"i2v:{i2v}")

idf_methods = {
    "log": lambda x: 1 + np.log(len(docs)) / (x + 1),
    "prob": lambda x: np.maximum(0, np.log(len(docs) - x) / (x + 1)),
    "len_norm": lambda x: x / (np.sum(np.square(x) + 1))
}

tf_methods = {
    "log": lambda x: np.log(1 + x),
    "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),  # 按行相加，并且保持其二维特性
    "boolean": lambda x: np.minimum(x, 1),
    "log_avg": lambda x: (1 + np.log(x)) / (1 + np.log(np.mean(x, axis=1, keepdims=True)))
}

# IDF表是一個所有詞語的重要程度表
def get_idf(method="log"):
    # inverse document frequency: low idf for a word appears in more docs, mean less important
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count
    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)  # [n_vocab, 1]

def get_tf(method="log"):
    # term frequency: how frequent a word appears in a doc
    _tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)  # [n_vocab, n_docs]
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        for v in counter.keys():
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]
    print(_tf[0])
    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)  # [n_vocab, n_doc]

def cosine_similarity(q: str, _tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))  # sqrt 開根號  square 平方
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity

# def docs_score(q: str, len_norm: bool = False):
#     q_words = q.replace(",", "").split(" ")
#
#     # add unknown words
#     unknown_v = 0
#     for v in set(q_words):
#         if v not in v2i:
#             v2i[v] = len(v2i)
#             i2v[len(v2i) - 1] = v
#             unknown_v += 1
#
#     if unknown_v > 0:
#         _idf = np.concatenate((idf, np.zeros((unknown_v, 1))), axis=0)  # 陣列拼接
#         _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
#     else:
#         _idf, _tf_idf = idf, tf_idf
#     counter = Counter(q_words)
#     q_tf = np.zeros((len(_idf), 1), dtype=np.float)  # [n_vocab, 1]
#     for v in counter.keys():
#         q_tf[v2i[v], 0] = counter[v]
#
#     q_vec = q_tf * _idf  # [n_vocab, 1]
#     q_scores = cosine_similarity(q_vec, _tf_idf)
#     if len_norm:
#         len_docs = [len(d) for d in docs_words]
#         q_scores = q_scores / np.array(len_docs)
#     return q_scores
#
# def get_keywords(n=2):
#     for c in range(3):
#         col = tf_idf[:, c]
#         idx = np.argsort(col)[-n:]
#         print(f"doc={c}, top={n} keywords={[i2v[i] for i in idx]}")

def show_tf_idf(tf_idf, vocb, filename):
    # [n_vocab, n_doc]
    plt.imshow(tf_idf, cmap="YlGn", vmin=tf_idf.min(), vmax=tf_idf.max())
    plt.xticks(np.arange(tf_idf.shape[1]), vocb, fontsize=6, rotation=90)
    plt.yticks(np.arange(tf_idf.shape[0]), np.arange(1, tf_idf.shape[1] + 1), fontsize=6)
    plt.tight_layout()  # 圖表過度集中可以使用.tight_layout分開
    plt.savefig(f"{filename}.png", format="png", dpi=1200)
    plt.show()


tf = get_tf()  # [n_vocab, n_doc]
# idf = get_idf()  # [n_vocab, 1]
# tf_idf = tf * idf   # [n_vocab, n_doc]
# print("tf shape(vecb in each docs): ", tf.shape)
# print("tf samples:", tf[:2])
# print("idf shape(vecb in all docs): ", idf.shape)
# print("idf samples:", idf[:2])
# print("tf_idf shape: ", tf_idf.shape)
# print("tf_idf sample:", tf_idf[:2])

# test
# get_keywords()
# q = "I get a coffee cup"
# scores = docs_score(q)
# print(f"scores = {scores}")
# d_ids = scores.argsort()[-3:][::-1]
# print(f"top 3 docs for {q}: {[docs[i] for i in d_ids]}")

# show_tf_idf(tf_idf.T, [i2v[i] for i in range(len(i2v))], "tf_idf_matrix")
