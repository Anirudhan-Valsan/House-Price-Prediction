from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained XGBoost model
model_filename = "model\PKHPP_model.pkl"
xgb_model = joblib.load(model_filename)

# Load the pre-fitted ColumnTransformer
ct_filename = "model\column_transformer.pkl"  # Assuming you saved your ColumnTransformer
ct = joblib.load(ct_filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        property_type = request.form['property_type']
        location = request.form['location']
        city = request.form['city']
        baths = int(request.form['baths'])
        purpose = request.form['purpose']
        bedrooms = int(request.form['bedrooms'])
        area_in_marla = float(request.form['area_in_marla'])

        new_input_data = pd.DataFrame(
            [[property_type, location, city, baths, purpose, bedrooms, area_in_marla]],
            columns=['property_type', 'location', 'city', 'baths', 'purpose', 'bedrooms', 'Area_in_Marla']
        )

        # Transform the new input data using the already fitted ColumnTransformer
        new_input_data_encoded = ct.transform(new_input_data)

        # Predict using the trained XGBoost model
        predicted_price = xgb_model.predict(new_input_data_encoded)

        return render_template('result.html', predicted_price=predicted_price[0])


if __name__ == '__main__':
    app.run(debug=True,port=8080)
