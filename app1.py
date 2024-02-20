from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved scaler and models
loaded_scaler = joblib.load('scaler.pkl')
loaded_dt_model = joblib.load('decision_tree_model.pkl')
loaded_lr_model = joblib.load('linear_regression_model.pkl')
loaded_rf_model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        open_input = float(request.form['open'])
        high_input = float(request.form['high'])
        low_input = float(request.form['low'])
        adj_close_input = float(request.form['adj_close'])
        volume_input = float(request.form['volume'])

        # Normalize user inputs using the loaded scaler
        user_inputs_normalized = loaded_scaler.transform(
            np.array([[open_input, high_input, low_input, adj_close_input, volume_input]])
        )

        # Make predictions using the loaded models
        dt_prediction = loaded_dt_model.predict(user_inputs_normalized)[0]
        lr_prediction = loaded_lr_model.predict(user_inputs_normalized)[0]
        rf_prediction = loaded_rf_model.predict(user_inputs_normalized)[0]

        # Determine the best prediction
        best_model = min([('Decision Tree', dt_prediction),
                          ('Linear Regression', lr_prediction),
                          ('Random Forest', rf_prediction)],
                         key=lambda x: x[1])

        return render_template('result.html', prediction=best_model[1], model=best_model[0])

if __name__ == '__main__':
    app.run(debug=True)
