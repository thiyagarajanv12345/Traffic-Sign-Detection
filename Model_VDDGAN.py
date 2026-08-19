import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from Evaluation_All import dehazing_metrics


def sampling(mu, log_var):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps


def VC_DDGAN_Generator_Model():
    inp = layers.Input(shape=(512, 512, 3))

    # Encoder
    x = layers.Conv2D(64, 4, 2, padding='same')(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)

    mu = layers.Dense(256)(x)
    log_var = layers.Dense(256)(x)

    z = layers.Lambda(lambda t: sampling(t[0], t[1]))([mu, log_var])

    # Decoder
    x = layers.Dense(64 * 64 * 256)(z)
    x = layers.Reshape((64, 64, 256))(x)

    x = layers.Conv2DTranspose(128, 4, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out = layers.Conv2DTranspose(3, 4, 2, padding='same', activation='sigmoid')(x)

    model = models.Model(inp, out)
    return model


def Dehaze_Generator(Image):
    model = VC_DDGAN_Generator_Model()

    # Preprocess
    img = Image.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    out = model.predict(img, verbose=0)

    # Postprocess
    out = (out[0] * 255).astype(np.uint8)
    Eval = dehazing_metrics(Image, out)
    return Eval, out

