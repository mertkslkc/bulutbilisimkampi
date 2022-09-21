from tensorflow import keras
import tensorflow as tf
import numpy as np
from numpy.ma.core import size
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

#/opt/app-root/src/bulutbilisimkampi/

model_dir = 'model.h5'
model = keras.models.load_model(model_dir)

anomaly = []

def predict(data):
    anomaly.clear()
    df = pd.Series(data)
    value = np.array(data["cpu_usage"])
    value = value.reshape((4,3,1))
    result = model.predict(value,verbose=0)
    loss = np.mean(np.abs(result - value), axis=1)
    threshold = np.mean(loss)
    df['threshold'] = threshold

    for i in range(len(df['cpu_usage'])):
        anomaly.append(bool(df['cpu_usage'][i] > threshold))
    df['anomaly'] = anomaly

    return  {"datetime": list(df['datetime']), "cpu_usage": list(df['cpu_usage']), "anomaly": df['anomaly']}
