"""
demo.py

Gradio Interface for Used Cars Price Prediction
"""

import os

import gradio as gr

from src.inference import predict_price


# ============================================================
# Prediction Wrapper
# ============================================================

def predict_car_price(
    location,
    brand,
    year,
    kilometers,
    fuel_type,
    transmission,
    owner_type,
    mileage,
    engine,
    power,
    seats,
):
    """
    Wrapper function for Gradio.
    """

    car_data = {

        "Location": location,

        "Year": int(year),

        "Kilometers_Driven": int(kilometers),

        "Fuel_Type": fuel_type,

        "Transmission": transmission,

        "Owner_Type": owner_type,

        "Mileage": float(mileage),

        "Engine": float(engine),

        "Power": float(power),

        "Seats": int(seats),

        "Brand": brand,

        "Car_Age": 2026 - int(year),
    }


    price = predict_price(car_data)


    return f"₹ {price:.2f} Lakhs"


# ============================================================
# Gradio Interface
# ============================================================


demo = gr.Interface(

    fn=predict_car_price,


    inputs=[

        gr.Textbox(
            label="Location",
            value="Mumbai"
        ),

        gr.Textbox(
            label="Brand",
            value="Maruti"
        ),

        gr.Number(
            label="Manufacturing Year",
            value=2018
        ),

        gr.Number(
            label="Kilometers Driven",
            value=42000
        ),


        gr.Dropdown(
            [
                "Petrol",
                "Diesel",
                "CNG"
            ],
            label="Fuel Type",
            value="Diesel"
        ),


        gr.Dropdown(
            [
                "Manual",
                "Automatic"
            ],
            label="Transmission",
            value="Manual"
        ),


        gr.Dropdown(
            [
                "First",
                "Second",
                "Third",
                "Fourth & Above"
            ],
            label="Owner Type",
            value="First"
        ),


        gr.Number(
            label="Mileage",
            value=21.4
        ),


        gr.Number(
            label="Engine (CC)",
            value=1248
        ),


        gr.Number(
            label="Power (bhp)",
            value=88.5
        ),


        gr.Number(
            label="Seats",
            value=5
        ),

    ],


    outputs=gr.Textbox(
        label="Estimated Price"
    ),


    title="🚗 Used Cars Price Prediction",


    description=
    """
    ML-powered used car valuation system
    built using XGBoost + FastAPI + Gradio.
    """
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)