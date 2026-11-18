import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8],
    'Education': [10, 12, 14, 16, 12, 14, 16, 18],
    'Age': [22, 24, 26, 28, 30, 32, 34, 36],
    'Salary': [20000, 25000, 30000, 35000,
               40000, 45000, 50000, 55000]
}

df = pd.DataFrame(data)

# Independent and Dependent Variables
X = df[['Experience', 'Education', 'Age']]
y = df['Salary']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Multiple Linear Regression")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted Salary:", y_pred)
print("Actual Salary:", list(y_test))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
