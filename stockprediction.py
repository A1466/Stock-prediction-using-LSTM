#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


# In[26]:


#Load the local .csv file as dataframe
data = pd.read_csv('reliance.csv')
print(data)


# In[27]:


#Extract the 'Close' prices as the target variable
data = data[['Date', 'Close']]  # Extract Date and Close columns
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime
data.set_index('Date', inplace=True)  # Set 'Date' as the index
data.sort_index(inplace=True)  # Sort by date
target_data = data['Close'].values.reshape(-1, 1)  # Extract 'Close' prices as target variable


# In[28]:


#Data visualization
data.plot()
plt.show()


# In[29]:


#Normalize the target data
scaler = MinMaxScaler()
target_data = scaler.fit_transform(target_data)


# In[30]:


#Split the data into training and testing sets
train_size = int(len(target_data) * 0.8)
train_data = target_data[:train_size]
test_data = target_data[train_size:]


# In[31]:


# Create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 10  # You can adjust this sequence length
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)


# In[32]:


# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')


# In[33]:


# Train the model
model.fit(train_sequences, train_data[seq_length:], epochs=50, batch_size=64)


# In[34]:


#Make predictions on the test set
predicted = model.predict(test_sequences)


# In[35]:


# Inverse transform the predictions to get actual stock prices
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(test_data[seq_length:])


# In[36]:


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size+seq_length:], actual_prices, label='Actual Prices')
plt.plot(data.index[train_size+seq_length:], predicted_prices, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




