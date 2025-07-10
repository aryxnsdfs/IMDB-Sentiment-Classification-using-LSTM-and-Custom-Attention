import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Layer
import numpy as np

vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(weights * inputs, axis=1)
        return context_vector

input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(vocab_size, 128)(input_layer)
lstm_output = LSTM(64, return_sequences=True)(embedding_layer)
attention_output = AttentionLayer()(lstm_output)
output_layer = Dense(1, activation='sigmoid')(attention_output)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_data=(x_test, y_test))
