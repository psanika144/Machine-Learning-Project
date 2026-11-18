from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Create Model
model = DecisionTreeClassifier(random_state=42)

# Step 4: Train Model
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("Machine Learning Modelling Process")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
print("Predictions:", y_pred)
print("Actual Values:", y_test)
print("Model Accuracy:", accuracy)
