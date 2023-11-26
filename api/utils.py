import tensorflow as tf


LATENT_DIM = 256
BATCH_SIZE = 1

def infer(model, image, latent_vector=None):
    if latent_vector is None:
        latent_vector = tf.random.normal(
            shape=(BATCH_SIZE, LATENT_DIM), mean=0.0, stddev=2.0
        )
    image = tf.stack([image])
    prediction = model.predict([latent_vector, image])
    prediction = (127.5 * (prediction + 1)).astype(int)
    return prediction
