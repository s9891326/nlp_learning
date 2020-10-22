from tensorflow import keras
import tensorflow as tf

class CBOW(keras.Model):
    def __init__(self, v_dim, emb_dim):
        super().__init__()
        self.embeddings = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  #[n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.1)
        )
        self.nce_w = self.add_weight(
            name="nce_w", shape=[v_dim, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 1.))  # [n_vocab, emb_dim]
        self.nce_b = self.add_weight(
            name="nce_b", shape=(v_dim, ),
            initializer=keras.initializers.Constant(0., 1.))  # [n_vocab, ]

        self.opt = keras.optimizers.Adam(0.01)

    def call(self, inputs, training=None, mask=None):
        # x.shape = [n, skip_window * 2]
        o = self.embeddings(inputs)  # [n, skip_window*2, emb_dim]
        o = tf.reduce_mean(o, axis=1)  # [n, emb_dim]
        return o

    # negative sampling: take one positive label and num_sampled negative labels to compute the loss
    # in order to reduce the computation of full softmax
