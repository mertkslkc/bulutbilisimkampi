from tensorflow import keras
import tensorflow as tf
import numpy as np
from numpy.ma.core import size
import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

model_dir = '/opt/app-root/src/bulutbilisimkampi/model.h5'
model = keras.models.load_model(model_dir)
anomaly = []

def predict(data):    
    value = np.array(data["cpu_usage"])
    value = value.reshape((4,3,1))
    result = model.predict(value,verbose=0)
    loss = np.mean(np.abs(result - value), axis=1)
    threshold = np.mean(loss)
    
    df = pd.Series(data)
    for i in range(len(df['cpu_usage'])):
        anomaly.append(df['cpu_usage'][i] > threshold)
    df['anomaly'] = anomaly
    return  {"datetime": str(df['datetime']), "cpu_usage": str(df['cpu_usage']), "anomaly": str(df['anomaly'])}
