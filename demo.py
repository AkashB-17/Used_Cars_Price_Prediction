from src.inference import predict_price

sample = {
    "Location": "Mumbai",
    "Year": 2018,
    "Kilometers_Driven": 42000,
    "Fuel_Type": "Diesel",
    "Transmission": "Manual",
    "Owner_Type": "First",
    "Mileage": 21.4,
    "Engine": 1248,
    "Power": 88.5,
    "Seats": 5,
    "Brand": "Maruti",
    "Car_Age": 8,
}

price = predict_price(sample)

print(f"Predicted Price: ₹{price:.2f} Lakhs")