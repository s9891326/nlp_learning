import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_addons as tfa
from mofanpy import utils


class CNNTranslation(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units,
                 max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units

        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1)
        )
        self.conv2ds = [
            keras.layers.Conv2D(16, (n, emb_dim), padding="valid", activation=keras.activations.relu)
            for n in range(2, 5)
        ]
        self.max_pools = [keras.layers.MaxPool2D((n, 1)) for n in [7, 6, 5]]
        self.encoder = keras.layers.Dense(units=units, activation=keras.activations.relu)

        # decoder
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1)
        )
        self.decoder_cell = keras.layers.LSTMCell(units=units)
        decoder_dense = keras.layers.Dense(dec_v_dim)

        # train decoder
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),
            output_layer=decoder_dense
        )

        # predict decoder
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),
            output_layer=decoder_dense
        )

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.01)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        embedded = self.enc_embeddings(x)  # [n, step, emb]
        o = tf.expand_dims(embedded, axis=3)  # [n, step=8, emb=16, 1] 在第axis位置增加一個維度
        co = [conv2d(o) for conv2d in self.conv2ds]  # [n, 7, 1, 16], [n, 6, 1, 16], [n, 5, 1, 16]
        # co = [self.max_pools[i](co[i]) for i in range(len(co))]  #[n, 1, 1, 16] * 3
        co = [max_pools(co[i]) for i, max_pools in enumerate(self.max_pools)]  #[n, 1, 1, 16] * 3
        co = [tf.squeeze(c, axis=[1, 2]) for c in co]  # [n, 16] * 3 從tensor中刪除所有大小是1的維度
        o = tf.concat(co, axis=1)  # [n, 16 * 3]  在co上面的第二個維度進行合併
        # t1 = [[1, 2, 3], [4, 5, 6]]
        # t2 = [[7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], axis=1) -> [[ 1,  2,  3,  7,  8,  9], [ 4,  5,  6, 10, 11, 12]]
        h = self.encoder(o)  # [n, units]
        return [h, h]

    def inference(self, x):
        s = self.encode(x)
        done, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s
        )
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False
            )
            pred_id[:, l] = o.sample_id
        return pred_id

    def train_logits(self, x, y, seq_len):
        s = self.encode(x)
        dec_in = y[:, :-1]  # ignore<go>
        dec_emb_in = self.dec_embeddings(dec_in)
        o, _, _ = self.decoder_train(dec_emb_in, s, sequence_length=seq_len)
        logits = o.rnn_output
        return logits

    def step(self, x, y, seq_len):
        with tf.GradientTape() as tape:
            logits = self.train_logits(x, y, seq_len)
            dec_out = y[:, 1:]
            loss = self.cross_entropy(dec_out, logits)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


def train():
    # get and process data
    data = utils.DateData(4000)
    print(f"Chinese  time order: yy/mm/dd {data.date_cn[:3]}\n English time order: dd/M/yyyy {data.date_en[:3]}")
    print(f"vocabularies: {data.vocab}")
    print(f"value to index: {data.v2i}")
    print(f"index to value: {data.i2v}")
    print(f"x index sample: \n{data.idx2str(data.x[0])}\n{data.x[0]}\n"
          f"y index sample: \n{data.idx2str(data.y[0])}\n{data.y[0]}")

    model = CNNTranslation(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token
    )

    # training
    for t in range(1500):
        bx, by, decoder_len = data.sample(32)
        loss = model.step(bx, by, decoder_len)
        if t % 100 == 0:
            target = data.idx2str(by[0, 1:-1])
            pred = model.inference(bx[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(bx[0])
            print(f"t: {t}"
                  f"| loss: {loss:.3f}"
                  f"| input: {src}"
                  f"| target: {target}"
                  f"| inference: {res}"
                  )

if __name__ == '__main__':
    train()
