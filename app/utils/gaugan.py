import numpy as np
import requests
import streamlit as st
import sys
import tensorflow as tf

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from settings import GAUGAN_PREDICT_URL


@st.cache_resource
def load_model():
    gaugan = tf.keras.models.load_model('./models/gaugan')
    return gaugan

def prepare_image(image):
    image = tf.image.resize(image, (256, 256))
    image = image[:,:,:3]


def infer(model, batch_size, latent_dim, image):
    latent_vector = tf.random.normal(
        shape=(batch_size, latent_dim), mean=0.0, stddev=2.0
    )
    image = tf.stack(image)
    prediction = model.predict([latent_vector, image])
    prediction = (127.5 * (prediction + 1)).astype(int)
    return prediction


def api_prediction(image, latent_vector=None):
    response = requests.post(
        GAUGAN_PREDICT_URL, 
        json={'image': image.tolist()}, 
        headers={"Content-Type": "application/json"}
    )
    image = np.array(response.json().get('prediction'))
    return image