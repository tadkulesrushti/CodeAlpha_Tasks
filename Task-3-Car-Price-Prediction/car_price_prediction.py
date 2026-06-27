# Car Price Prediction using Linear Regression
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("car data.csv")

# Display the first five rows
print("First 5 Rows of Dataset:\n")
print(df.head())

# Display dataset information
print("\n-----------------------------")
print(df.info())

# Display dataset shape
print("\n-----------------------------")
print("Dataset Shape:", df.shape)

# Display unique values of categorical columns
print("\nFuel Types:")
print(df["Fuel_Type"].unique())

print("\nSelling Types:")
print(df["Selling_type"].unique())

print("\nTransmission Types:")
print(df["Transmission"].unique())

# Encode categorical features
df.replace({
    "Fuel_Type": {
        "Petrol": 0,
        "Diesel": 1,
        "CNG": 2
    },
    "Selling_type": {
        "Dealer": 0,
        "Individual": 1
    },
    "Transmission": {
        "Manual": 0,
        "Automatic": 1
    }
}, inplace=True)

# Display encoded dataset
print("\nDataset after Encoding:\n")
print(df.head())

# Split Features and Target Variable
X = df.drop(["Car_Name", "Selling_Price"], axis=1)
y = df["Selling_Price"]

# Split data into Training and Testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train the Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# Evaluate the Model

# Training Score
train_prediction = model.predict(X_train)
train_r2 = r2_score(y_train, train_prediction)

# Testing Score
test_prediction = model.predict(X_test)
test_r2 = r2_score(y_test, test_prediction)

print("\nTraining R2 Score:", train_r2)
print("Testing R2 Score:", test_r2)

# Visualize Actual vs Predicted Prices

plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

# Predict Selling Price for a New Car
new_car = pd.DataFrame({
    "Year": [2020],
    "Present_Price": [8.50],
    "Driven_kms": [15000],
    "Fuel_Type": [0],
    "Selling_type": [0],
    "Transmission": [0],
    "Owner": [0]
})

predicted_price = model.predict(new_car)

print("\nPredicted Selling Price: {:.2f} Lakh".format(predicted_price[0]))