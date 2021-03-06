import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_addons as tfa
from mofanpy import utils


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units,
                 max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units

        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1)
        )
        self.encoder = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)

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
        embedded = self.enc_embeddings(x)
        init_s = [tf.zeros((x.shape[0], self.units)), tf.zeros((x.shape[0], self.units))]
        o, h, c = self.encoder(embedded, initial_state=init_s)
        return [h, c]  # state

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

    model = Seq2Seq(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token
    )

    # training
    for t in range(1500):
        bx, by, decoder_len = data.sample(32)
        print(f"bx: {bx}, by: {by}, decoder_len: {decoder_len}")
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
