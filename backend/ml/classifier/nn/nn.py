import numpy as np
import tensorflow as tf
from tensorflow import keras

'''
*TO-DO LIST*
    - implement regularisation
'''


class Network(tf.keras.Model):
    def __init__(self, shape, activation, outputActivation):
        super().__init__()
        self.nodes = []
        self.nodes.append(keras.layers.Flatten(input_shape=shape[0]))
        for density in shape[1:-1]:
            self.nodes.append(tf.keras.layers.Dense(density, activation=activation))
        self.nodes.append(tf.keras.layers.Dense(shape[-1], activation=outputActivation))

    def call(self, inputs, training=False):
        x = self.nodes[0](inputs)
        for layer in self.nodes[1:-1]:
            x = layer(x)
        return self.nodes[-1](x)
    
    def train(self, learningRate, lossFunction, epochs, batchSize, validationSplit):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        opt = keras.optimizers.Adam(learning_rate=learningRate)
        self.compile(opt, loss=lossFunction, metrics=['accuracy'])
        self.fit(x_train, y_train, epochs=epochs, batch_size=batchSize, validation_split=validationSplit) 


# TESTING
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# testModel = Network([[28, 28], 256, 128, 10], "relu", "softmax")
# testModel.train(0.001, 'sparse_categorical_crossentropy', 5, 32, 0.001)
# predictions = testModel.predict(x_test[:10])  # Predict the first 10 test samples

# # Convert probabilities to predicted digit
# predicted_digits = np.argmax(predictions, axis=1)
# print("Predicted digits:", predicted_digits)
# print("Answers:", y_test[:10])