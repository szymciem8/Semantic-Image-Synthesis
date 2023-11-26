import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from utils import infer

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

gaugan_model = tf.keras.models.load_model('./models/gaugan')

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/api/v1/generative/models/gaugan/predict/")
async def gaugan_predict(request: Request):
    data = await request.json()
    image = data.get('image')
    latent_vector = data.get('latent_vector')
    prediction = infer(gaugan_model, image, latent_vector)
    return {'prediction': prediction.tolist()}