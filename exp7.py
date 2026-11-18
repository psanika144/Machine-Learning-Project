from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LogisticRegression(max_iter=200)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Logistic Regression")
print("Predicted Values:", y_pred)
print("Actual Values:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# New Prediction
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_data)

print("New Prediction:", iris.target_names[prediction[0]])
