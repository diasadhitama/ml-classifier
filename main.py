# Import Library

import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

# Create Object
app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Endpoint
@app.get('/')
def index():
    return{"message":'Selamat Datang'}

@app.get('/{nama}')
def get_nama(nama:str):
    return{'nama panjang': f'{nama}'}

@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    # print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if (prediction[0] > 0.5):
        prediction = 'Fake Note'
    else:
        prediction = 'Its a Bank note'
    return{
        'prediction': prediction
    }

    # Konfigurasi Server
    if __name__ == '__main__':
        uvicorn.run(app, host='127.0.0.1', port=8000)