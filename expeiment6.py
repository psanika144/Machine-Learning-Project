import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Model
model = KNeighborsRegressor(n_neighbors=3)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("K-Nearest Neighbor Regression")
print("Predicted Values:", y_pred)
print("Actual Values:", y_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
