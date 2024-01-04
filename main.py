from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI();

# Set up allowed origins, methods, and headers
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    data: List[int]


with open('lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as s:
    sc = pickle.load(s)


@app.post('/ml/predict')
async def predict_endpoint(item: Item):
    data = item.data
    X_new = np.array([data])
    X_new = sc.transform(X_new)
    y_new = model.predict(X_new[:, 1:])
    X_new[0][10] = y_new
    temp_sc = sc.inverse_transform(X_new)
    temp_sc = np.round(temp_sc)
    ans =  temp_sc[0][-1]
    return {"Prediction": ans}