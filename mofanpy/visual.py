import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.pyplot import cm
import os
from mofanpy import utils

def show_tfidf(tfidf, vocb, filename):
    # [n_vocab, n_doc]
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(tfidf.shape[1]), vocb, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[1]+1), fontsize=6)
    plt.tight_layout()
    plt.savefig("./visual/results/%s.png" % filename, format="png", dpi=500)
    plt.show()


def show_w2v_word_embedding(model, data: utils.Dataset, path):
    word_emb = model.embeddings.get_weights()[0]
    for i in range(data.num_word):
        c = "blue"
        try:
            int(data.i2v[i])
        except ValueError:
            c = "red"
        plt.text(word_emb[i, 0], word_emb[i, 1], s=data.i2v[i], color=c, weight="bold")
    plt.xlim(word_emb[:, 0].min() - .5, word_emb[:, 0].max() + .5)
    plt.ylim(word_emb[:, 1].min() - .5, word_emb[:, 1].max() + .5)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("embedding dim1")
    plt.ylabel("embedding dim2")
    plt.savefig(path, dpi=300, format="png")
    plt.show()
