import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Define the training dataset
X_train = np.array([[28, 30, 29, 31, 27], [30, 28, 27, 29, 26], [29, 31, 32, 28, 30], [27, 29, 28, 30, 31]])
y_train = np.array([31, 27, 29, 28])

# Define the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Export the trained model as a .pkl file
with open('linear_regression.pkl', 'wb') as f:
    pickle.dump(model, f)
