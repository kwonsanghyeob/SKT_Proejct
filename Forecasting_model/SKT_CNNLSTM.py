import tensorflow as tf

class Mymodel_CNNLSTM(tf.keras.Model):
    def __init__(self):
        super(Mymodel_CNNLSTM,self).__init__()
        LSTM = tf.keras.layers.LSTM
        Dense = tf.keras.layers.Dense
        dropout = tf.keras.layers.Dropout
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

        self.sequence = list()
        self.sequence.append(LSTM(units = 256, return_sequences=True))
        self.sequence.append(LSTM(units= 256, return_sequences=False))
        self.sequence.append(dropout(0.2))
        self.sequence.append(Dense(24, activation='relu')) #5분단위 12*24
    def call(self, x, training = False):
        for layer in self.sequence:
            x = layer(x)
        return x