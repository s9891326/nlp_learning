from tensorflow import keras


class ELMo(keras.Model):
    def __init__(self):
        # encoder
        self.word_embed = keras.layers.Embedding(...)

        # forward lstm
        # self.fs = [keras.layers.LSTM(unit, return_sequences=True) for _ in range(n_layers)]
        # self.f_logits = keras.layers.Dense(v_dim)

        # backward lstm
        # self.bs = [keras.layers.LSTM(units, return_sequences=True, go_backwards=True) for _ in range(n_layers)]
        # self.b_logits = keras.layers.Dense(v_dim)

    def call(self, seqs):
        embedded = self.word_embed(seqs)

