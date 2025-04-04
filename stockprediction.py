# -------------------------------------
# 1. Import necessary libraries
# -------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# -------------------------------------
# 2. Download stock data
# -------------------------------------
ticker = 'BAJFINANCE.NS'
df = yf.download(ticker, period='5y')

# Rename 'Close' for consistency (optional)
df = df.rename(columns={'Close': 'Close'})

# -------------------------------------
# 3. Visualize the Close price
# -------------------------------------
plt.figure(figsize=(16, 6))
plt.plot(df['Close'], label='Close Price')
plt.title(f'{ticker} Closing Price History')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# -------------------------------------
# 4. Prepare the data
# -------------------------------------
data = df.filter(['Close'])
dataset = data.values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Training dataset length
train_len = math.ceil(len(scaled_data) * 0.8)

# Training data
train_data = scaled_data[0:train_len, :]

# Create training sequences
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM input: (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -------------------------------------
# 5. Build the LSTM Model
# -------------------------------------
model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=3)

# -------------------------------------
# 6. Prepare Test Data
# -------------------------------------
test_data = scaled_data[train_len - 60:, :]
x_test = []
y_test = dataset[train_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# -------------------------------------
# 7. Make Predictions
# -------------------------------------
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f'RMSE: {rmse:.2f}')

# -------------------------------------
# 8. Visualize the Results
# -------------------------------------
# Create new DataFrame for plotting
train = data[:train_len]
validation = data[train_len:].copy()
validation['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('LSTM Model: Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
