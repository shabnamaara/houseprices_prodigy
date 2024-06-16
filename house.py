import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(0)
n_samples = 100

square_footage = np.random.randint(500, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
prices = 50000 + (square_footage * 300) + (bedrooms * 10000) + (bathrooms * 5000) + np.random.randint(-20000, 20000, n_samples)

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})

# Split data into features and target variable
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Displaying the model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Example prediction
new_house = pd.DataFrame({
    'SquareFootage': [2500],
    'Bedrooms': [4],
    'Bathrooms': [3]
})
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")
