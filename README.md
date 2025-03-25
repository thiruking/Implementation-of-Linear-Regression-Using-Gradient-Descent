# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1️. Load and preprocess the dataset (convert categorical data, normalize features).
2️. Initialize model parameters (theta), learning rate (α), and iterations.
3️. Compute cost function using Mean Squared Error (MSE).
4️. Apply Gradient Descent to update theta iteratively.
5️. Use the trained model to predict profit for new input values.



## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: THIRUMALAI K
RegisterNumber: 212224240176 
*/
```
### Load and View the Dataset :
```
import pandas as pd

# Load dataset
df = pd.read_csv('50_Startups.csv')

# Display first 5 rows
df.head()

```
## OUTPUT :
![image](https://github.com/user-attachments/assets/1405c08d-bf68-4af1-bf27-85900c13d4ac)

### Check Dataset Information :

```
df.info()
df.describe()
```
## OUTPUT :
![image](https://github.com/user-attachments/assets/c605abec-d24b-49c8-a303-a16126c30db0)

### Convert Categorical Data (State Column) :

```
df = pd.get_dummies(df, columns=['State'], drop_first=True)
df.head()

```
## OUTPUT :

![image](https://github.com/user-attachments/assets/6388c03b-1fb8-4be9-abb5-0fdd2d640079)

### Split Data into X & y :

```
X = df.drop(columns=['Profit']).values  # Features
y = df['Profit'].values.reshape(-1, 1)  # Target Output

```
### Feature Scaling (Normalization) :
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

```

### Create Gradient Descent Function :

```
import numpy as np

# Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Gradient Descent Algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

```
### Train Model using Gradient Descent :

```
import matplotlib.pyplot as plt

# Add bias term (column of ones)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize theta
theta = np.zeros((X_b.shape[1], 1))

# Set hyperparameters
alpha = 0.01  # Learning rate
iterations = 1000

# Train model using gradient descent
theta_final, cost_history = gradient_descent(X_b, y, theta, alpha, iterations)

# Print final theta values
print("Final theta values:", theta_final)

# Plot cost function
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Iterations")
plt.show()

```
## OUTPUT :

![image](https://github.com/user-attachments/assets/88ab1d13-e749-46f8-bb82-0e724eb65b4d)

### Test the Model with New Data :

```
# Example Test Input (Modify values as needed)
test_data = np.array([[160000, 130000, 450000, 1, 0]])  

# Scale the test data
test_data_scaled = scaler.transform(test_data)

# Add bias term
test_data_scaled_b = np.c_[np.ones((test_data_scaled.shape[0], 1)), test_data_scaled]

# Predict profit
predicted_profit = test_data_scaled_b.dot(theta_final)
print("Predicted Profit:", predicted_profit[0][0])

```
## OUTPUT :

![image](https://github.com/user-attachments/assets/83db081f-eca4-4001-af84-834fef0d8c20)



## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
