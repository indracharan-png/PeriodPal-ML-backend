import pickle
import numpy as np

# Load the trained model from the .pkl file
with open('lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as s:
    sc = pickle.load(s)

# Use the model to make a prediction
X_new = np.array([[31, 30, 29, 31, 30, 31, 30, 29, 31, 10, 20]])

# print(model)
X_new = sc.transform(X_new)
print(X_new)


y_new = model.predict(X_new[:, 1:])
print(y_new)

X_new[0][10] = y_new
print(X_new)

temp_sc = sc.inverse_transform(X_new)
print(temp_sc)

temp_sc = np.round(temp_sc)
# print(y_new)
print("Predicted length of next cycle:", temp_sc[0][-1])
