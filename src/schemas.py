"""
schemas.py

Pydantic schemas for request and response models.
"""

from pydantic import BaseModel


class CarFeatures(BaseModel):
    Location: str
    Year: int
    Kilometers_Driven: int
    Fuel_Type: str
    Transmission: str
    Owner_Type: str
    Mileage: float
    Engine: float
    Power: float
    Seats: int
    Brand: str
    Car_Age: int


class PredictionResponse(BaseModel):
    predicted_price: float