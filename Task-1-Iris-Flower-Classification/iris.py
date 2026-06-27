import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Iris.csv")

# Features and Target
X = data.iloc[:, 1:5]
y = data.iloc[:, 5]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nFirst 5 Predictions:")
print(model.predict(X_test[:5]))

print("\nActual Values:")
print(y_test.iloc[:5].values)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

import matplotlib.pyplot as plt

data["Species"].value_counts().plot(kind="bar")
plt.title("Iris Flower Count")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()