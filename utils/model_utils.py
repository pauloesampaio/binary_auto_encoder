import numpy as np
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    Activation,
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def build_model(input_size, latent_dimension, n_filters, filter_sizes, strides):
    """Builds auto encoder model

    Args:
        input_size (int): Image input size
        latent_dimension (int): Dimensionality of latent space
        n_filters (list): List with number of units by layer
        filter_sizes (list): List with kernel size of each layer
        strides (list): List with stride of each layer

    Returns:
        keras.Model, keras.Model, keras.Model: Encoder, Decoder and Autoencoder keras models
    """
    input_img = Input(shape=(input_size, input_size, 1))
    x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding="same")(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)
    for n_units, k_size, stride in zip(n_filters[1:], filter_sizes[1:], strides[1:]):
        x = Conv2D(n_units, kernel_size=k_size, strides=stride, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=-1)(x)
    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dimension)(x)
    encoder = Model(input_img, latent, name="encoder")

    latentInputs = Input(shape=(latent_dimension,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    for n_units, k_size, stride in zip(
        n_filters[::-1], filter_sizes[::-1], strides[::-1]
    ):
        x = Conv2DTranspose(n_units, k_size, strides=stride, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=-1)(x)
    x = Conv2DTranspose(1, (input_size, input_size), padding="same")(x)
    outputs = Activation("sigmoid")(x)
    decoder = Model(latentInputs, outputs, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder
