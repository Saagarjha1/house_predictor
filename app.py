from flask import Flask, render_template, request
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load the dataset from the CSV file
file_path = '/workspaces/codespaces-flask/Housing.csv'
df = pd.read_csv(file_path)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'])

# Separate features and target variable
target_column_name = 'price'
X = df.drop(target_column_name, axis=1)
y = df[target_column_name]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize histogram gradient boosting regressor
hist_gb_regressor = HistGradientBoostingRegressor(max_iter=1000, max_depth=3, learning_rate=0.1)

# Train the regressor
hist_gb_regressor.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the form data
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = request.form['mainroad']
        guestroom = request.form['guestroom']
        basement = request.form['basement']
        hotwaterheating = request.form['hotwaterheating']
        airconditioning = request.form['airconditioning']
        parking = int(request.form['parking'])
        prefarea = request.form['prefarea']
        furnishingstatus = request.form['furnishingstatus']

        # Convert categorical values to one-hot encoding
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad_yes': 1 if mainroad == 'yes' else 0,
            'guestroom_yes': 1 if guestroom == 'yes' else 0,
            'basement_yes': 1 if basement == 'yes' else 0,
            'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
            'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
            'parking': parking,
            'prefarea_yes': 1 if prefarea == 'yes' else 0,
            'furnishingstatus_furnished': 1 if furnishingstatus == 'furnished' else 0,
            'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
            'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame(columns=X.columns, data=[input_data])

        # Make prediction
        predicted_price = hist_gb_regressor.predict(input_df)[0]

        return render_template('index.html', predicted_price=predicted_price)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
