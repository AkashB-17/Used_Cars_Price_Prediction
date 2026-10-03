"""
demo.py

Gradio UI for the Used Cars Price Prediction model.
"""

import os

import gradio as gr

from src.inference import predict_price

LOCATIONS = [
    "Ahmedabad", "Bangalore", "Chennai", "Coimbatore", "Delhi",
    "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune",
]

BRANDS = [
    "Ambassador", "Audi", "Bentley", "BMW", "Chevrolet", "Datsun", "Fiat",
    "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep",
    "Lamborghini", "Land", "Mahindra", "Maruti", "Mercedes-Benz", "Mini",
    "Mitsubishi", "Nissan", "Porsche", "Renault", "Skoda", "Smart",
    "Tata", "Toyota", "Volkswagen", "Volvo",
]

CURRENT_YEAR = 2026


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
    """Wrapper function for Gradio — builds the feature dict and formats the result."""

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
        "Car_Age": CURRENT_YEAR - int(year),
    }

    price = predict_price(car_data)

    return f"""
    <div class="result-box">
        <h2>₹ {price:.2f} Lakh</h2>
        <p>Estimated resale value based on current market trends</p>
    </div>
    """


# ============================================================
# Gradio Interface
# ============================================================

THEME = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="*shadow_drop_lg",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

css = """
.gradio-container { max-width: 1100px !important; }
.banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.banner h1 { margin: 0 0 10px 0; color: white; font-weight: 700; font-size: 28px; }
.banner p { margin: 0; color: #94a3b8; font-size: 16px; }
.result-box {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    margin-top: 20px;
}
.result-box h2 { margin: 0; color: white; font-weight: 700; font-size: 36px; }
.result-box p { margin: 10px 0 0 0; color: #d1fae5; font-size: 16px; }
"""

with gr.Blocks(theme=THEME, css=css, title="Used Cars Price Prediction") as demo:
    gr.HTML('''
        <div class="banner">
            <h1>🚗 Used Cars Price Prediction</h1>
            <p>Predicts a fair resale price for a used car from its specs — a mini version of the dynamic-pricing problem platforms solve at scale.</p>
            <p style="font-size: 14px; margin-top: 10px; color: #64748b;">Model: <strong>Random Forest</strong> &middot; R² = 0.89 &middot; MAE &approx; ₹1.49 Lakh (trained on ~6,000 listings)</p>
        </div>
    ''')

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📍 Location & Identity")
                location = gr.Dropdown(choices=LOCATIONS, value="Mumbai", label="Location")
                brand = gr.Dropdown(choices=BRANDS, value="Maruti", label="Brand")
                year = gr.Slider(1998, CURRENT_YEAR, value=2018, step=1, label="Year of Manufacture")

            with gr.Group():
                gr.Markdown("### 🛣️ Usage")
                kilometers = gr.Number(value=42000, label="Kilometers Driven", precision=0)
                owner_type = gr.Dropdown(
                    choices=["First", "Second", "Third", "Fourth & Above"],
                    value="First",
                    label="Owner Type",
                )

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Engine & Performance")
                fuel_type = gr.Dropdown(
                    choices=["Petrol", "Diesel", "CNG", "LPG", "Electric"],
                    value="Diesel",
                    label="Fuel Type",
                )
                transmission = gr.Dropdown(
                    choices=["Manual", "Automatic"], value="Manual", label="Transmission"
                )
                mileage = gr.Number(value=21.4, label="Mileage (kmpl or km/kg)")
                engine = gr.Number(value=1248, label="Engine (CC)")
                power = gr.Number(value=88.5, label="Power (bhp)")
                seats = gr.Slider(2, 10, value=5, step=1, label="Seats")

    predict_btn = gr.Button("Calculate Fair Price", variant="primary", size="lg")
    output = gr.HTML()

    predict_btn.click(
        fn=predict_car_price,
        inputs=[
            location, brand, year, kilometers, fuel_type,
            transmission, owner_type, mileage, engine, power, seats,
        ],
        outputs=output,
    )

    gr.Examples(
        examples=[
            ["Mumbai", "Maruti", 2018, 42000, "Diesel", "Manual", "First", 21.4, 1248, 88.5, 5],
            ["Delhi", "Hyundai", 2015, 60000, "Petrol", "Manual", "Second", 18.0, 1197, 82.0, 5],
            ["Bangalore", "Mercedes-Benz", 2019, 25000, "Diesel", "Automatic", "First", 14.0, 2143, 190.0, 5],
            ["Chennai", "Toyota", 2012, 90000, "Diesel", "Manual", "Third", 15.0, 2494, 100.0, 7],
        ],
        inputs=[
            location, brand, year, kilometers, fuel_type,
            transmission, owner_type, mileage, engine, power, seats,
        ],
        label="Try an example",
    )

    gr.Markdown(
        """
        ---
        Built with scikit-learn + FastAPI + Gradio · [Source](https://github.com/AkashB-17/Used_Cars_Price_Prediction)
        """
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
