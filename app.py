from fastapi import FastAPI

from src.inference import predict_price
from src.schemas import CarFeatures, PredictionResponse

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Used Cars Price Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures):

    price = predict_price(car.model_dump())

    return PredictionResponse(
        predicted_price=price
    )