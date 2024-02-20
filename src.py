import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data
data = pd.read_csv('tesla_stock.csv')  # Update the file path

# Select features (X) and target variable (y)
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
X = data[features]
y = data['Close']
# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
X_normalized = scaler.fit_transform(X)
scaler.feature_names_ = features  # Add this line to set feature names

# Save the scaler and models
joblib.dump(scaler, 'scaler.pkl')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Save the scaler and models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(lr_model, 'linear_regression_model.pkl')
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
sns.scatterplot(lr_model)
