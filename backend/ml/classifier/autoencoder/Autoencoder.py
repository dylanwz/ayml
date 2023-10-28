# Import ML-related libraries
import tensorflow as tf
from tensorflow import keras

# Import data-related libraries
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

plt.imshow(x_train[0], cmap="gray" )

# Format the input
input_layer = keras.Input(shape=(28,28,1), name="raw_input")
encoder_input = keras.layers.Flatten(name="encoder_in")(input_layer)

# Compress the input space into a lower-dimensional latent space... this becomes input for the decoder
# --> `f(x) = h`
encoder_output = keras.layers.Dense(64, activation="relu", name="encoder_out")(encoder_input)

# Attempt to reconstruct the original image from the encoded information in the latent space `h`
# --> `g(h) = x`
decoder_output = keras.layers.Dense(784, activation="relu", name="decoder_out")(encoder_output)

# Re-format the output
output_layer = keras.layers.Reshape((28,28,1), name="raw_output")(decoder_output)

# Save the encoder; the latent lower-dimensional space `h` is valuable
encoder = keras.Model(input_layer, encoder_output)

autoencoder = keras.Model(input_layer, output_layer, name="autoencoder")
autoencoder.summary() 

opt = keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(opt, loss="mse")
autoencoder.fit(x_train, x_train, epochs=3, batch_size=32, validation_split=0.1) 

choose_sample = 888
plt.imshow(x_test[choose_sample], cmap="gray" )

sample = autoencoder.predict([x_test[choose_sample].reshape(-1,28,28,1)])[0]
plt.imshow(sample, cmap="gray")

sample_latent = encoder.predict([x_test[choose_sample].reshape(-1,28,28)])[0] 
plt.imshow(sample_latent.reshape((8,8)), cmap="gray")