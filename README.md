# Used Cars Price Prediction

A dynamic pricing model for used cars — predicts a fair resale price (in Lakh INR) from a car's specs, and serves it through a FastAPI backend and a Gradio UI. Built as a mini version of the pricing-engine problem platforms like CARS24 solve at scale.

**Live demo:** https://used-cars-price-prediction-f645.onrender.com

> Note: hosted on Render's free tier, so the first request after inactivity may take ~30–60s to wake the instance.

## Problem

Every used car is different — age, mileage, engine, brand, and ownership history all affect resale value. This project trains and compares several regression models to predict that value, then exposes the best one through a clean API.

## Results

Trained on ~6,000 listings from a public Kaggle used-cars dataset. Models were trained on log-transformed price (to handle the right-skewed price distribution) but **scored on the actual price scale (Lakh INR)** so the metrics below are directly interpretable.

| Model | R² | MAE (Lakh) | RMSE (Lakh) |
|---|---|---|---|
| Linear Regression | 0.760 | 2.17 | 5.43 |
| **Random Forest (selected)** | **0.890** | **1.49** | **3.67** |
| XGBoost | 0.888 | 1.36 | 3.71 |
| CatBoost | 0.884 | 1.52 | 3.78 |

Random Forest was selected by R², though XGBoost has a slightly lower MAE — the three tree-based models are within noise of each other, and any of them would be a reasonable choice.

## Project Structure

```
├── app.py                  # FastAPI backend (entrypoint)
├── demo.py                 # Gradio UI
├── src/
│   ├── config.py           # Paths, constants, feature lists
│   ├── preprocessing.py    # Data cleaning + feature engineering
│   ├── training.py         # Model training + comparison
│   ├── inference.py        # Load model, run predictions
│   ├── schemas.py          # Pydantic request/response models
│   └── utils.py            # Shared helpers (logging, numeric cleaning)
├── data/raw/                # Raw dataset (used_cars_data.csv)
├── artifacts/                # Trained model + metrics (generated)
└── tests/
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Running the pipeline

```bash
# 1. Clean the raw data
uv run python -m src.preprocessing

# 2. Train and compare models (saves the best one to artifacts/)
uv run python -m src.training

# 3. Run the API
uv run uvicorn app:app --reload

# 4. Run the Gradio UI (separate from the API, calls the model directly)
uv run python demo.py
```

## API

**`GET /health`** — health check

**`POST /predict`** — predict a price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Location": "Mumbai",
    "Year": 2018,
    "Kilometers_Driven": 40000,
    "Fuel_Type": "Diesel",
    "Transmission": "Manual",
    "Owner_Type": "First",
    "Mileage": 18.0,
    "Engine": 1200,
    "Power": 85,
    "Seats": 5,
    "Brand": "Maruti",
    "Car_Age": 8
  }'
```

Response:
```json
{"predicted_price": 5.96}
```

Interactive API docs are available at `/docs` once the server is running.

## Data

Source: [Kaggle — Used Cars Price Prediction dataset](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction). Rows without a labeled price were dropped; unit-suffixed numeric fields (`Mileage`, `Engine`, `Power`) were parsed to plain numbers; `Brand` was extracted from the car `Name`; `Car_Age` was engineered from `Year`.

## Known limitations

- Predictions are only as good as the training data — the model hasn't seen recent (2024+) listings or brands/models outside the training set.
- `Car_Age` and `Brand` are currently required as direct API inputs rather than derived server-side from `Year`/`Name` — a caller needs to compute these before calling `/predict`.
