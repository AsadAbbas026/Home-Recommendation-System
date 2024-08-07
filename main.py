import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

n_samples = input("Enter any amount of random data samples: ")
bedrooms = np.random.randint(1, 6, size=int(n_samples))
bathrooms = np.random.randint(1, 5, size=int(n_samples))
square_footage = np.random.randint(800, 4001, size=int(n_samples))
locations = np.random.choice(['City A', 'City B', 'City C', 'City D'], size=int(n_samples))
amenities = np.random.choice(['None', 'Pool', 'Garage', 'Gym', 'Garden'], size=int(n_samples))
sale_prices = np.random.randint(100000, 1000001, size=int(n_samples))

data = pd.DataFrame({
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'SquareFootage': square_footage,
    'Location': locations,
    'Amenities': amenities,
    'SalePrice': sale_prices
})
filename = input("Enter the file name: ")
data.to_csv(filename + '.csv', index=False)

data = pd.read_csv(filename + '.csv')
data_encoded = pd.get_dummies(data, columns=["Location", "Amenities"])
print(data_encoded)

x = input("Enter the Feature You want to use for predicting the target: ")
X = data_encoded.drop(x, axis=1)
y = data_encoded['SalePrice']

n = input("Enter any number between 0 and 1: ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(n), random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(y_pred)
print('Mean Squared Error:', mse)