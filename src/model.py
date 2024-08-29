# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import shap
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv("data/deliverytime.txt")

# Calculate the distance between the restaurant and delivery location for each order
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

data['distance'] = data.apply(lambda row: haversine_distance(row['Restaurant_latitude'],
                                                             row['Restaurant_longitude'],
                                                             row['Delivery_location_latitude'],
                                                             row['Delivery_location_longitude']), axis=1)

# Prepare data for modeling
X = data[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]].values
y = data["Time_taken(min)"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

print(f'\nModel Performance:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r_squared:.2f}')

# Feature Importance using SHAP
rf_model = RandomForestRegressor()
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test.reshape(X_test.shape[0], -1))
shap.summary_plot(shap_values, X_test.reshape(X_test.shape[0], -1), feature_names=["Age", "Ratings", "Distance"])
