import ee
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

ee.Initialize(project='ee-arthurcourbevoie')
dataset = ee.ImageCollection('TOMS/MERGED')
start_date = '2024-01-01'
end_date = '2024-12-31'
filtered_dataset = dataset.filterDate(start_date, end_date)
roi = ee.Geometry.Rectangle([[-75, 59], [-20, 84]])

def extract_ozone(image):
    ozone_image = image.select('ozone')
    mean_ozone = ozone_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10000
    ).get('ozone')
    return image.set('mean_ozone', mean_ozone)

ozone_collection = filtered_dataset.map(extract_ozone)
data_list = ozone_collection.reduceColumns(ee.Reducer.toList(2), ['system:time_start', 'mean_ozone']).values().get(0).getInfo()

data = pd.DataFrame(data_list, columns=['time', 'ozone'])
data['time'] = pd.to_datetime(data['time'], unit='ms')
data = data.sort_values(by='time').reset_index(drop=True)
data = data.dropna()

data['day_of_year'] = data['time'].dt.dayofyear
data['month'] = data['time'].dt.month
data['year'] = data['time'].dt.year

features_to_scale = ['ozone', 'day_of_year', 'month']

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features_to_scale])
scaled_data = pd.DataFrame(scaled_features, columns=[f'{col}_scaled' for col in features_to_scale])

sequence_length = 30
def create_sequences(data, sequence_length):
    sequences_X = []
    sequences_y = []
    for i in range(len(data) - sequence_length):
        seq_x = data[['day_of_year_scaled', 'month_scaled']].iloc[i:i + sequence_length].values
        seq_y = data['ozone_scaled'].iloc[i + sequence_length]
        sequences_X.append(seq_x)
        sequences_y.append(seq_y)
    return np.array(sequences_X), np.array(sequences_y)

X, y = create_sequences(scaled_data, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)]

n_features = X_train.shape[2]
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

model = Sequential([
    Bidirectional(LSTM(128, activation='relu', return_sequences=True, input_shape=(sequence_length, n_features))),
    BatchNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(64, activation='relu')),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[rmse])

history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test), callbacks=callbacks)

model.save('Ozone_model_lstm.h5')

y_pred_scaled = model.predict(X_test_reshaped)

y_pred = scaler.inverse_transform(np.concatenate([y_pred_scaled, np.zeros_like(y_pred_scaled)], axis=1))[:, 0]
y_true = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))], axis=1))[:, 0]

t = data['time'].iloc[-len(y_test):].tolist()

plt.figure(figsize=(15, 6))
plt.plot(t, y_true, label='Actual Ozone Concentration')
plt.plot(t, y_pred, label='Predicted Ozone Concentration')
plt.xlabel('Datetime')
plt.ylabel('Ozone Concentration (mol/m²)')
plt.title('Actual vs. Predicted Ozone Concentration')
plt.legend()
plt.grid(True)
plt.show()

dates = data['time'].tolist()
ozone_values = data['ozone'].tolist()
df = pd.DataFrame({'date': dates, 'ozone': ozone_values})

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['ozone'], label='Actual Ozone Concentration')
plt.xlabel('Date')
plt.ylabel('Concentration d\'ozone (mol/m²)')
plt.title('Concentration d\'ozone au Greenland')
plt.grid(True)
plt.show()

csv_data = pd.DataFrame(data_list, columns=['time', 'ozone'])
csv_data['time'] = pd.to_datetime(csv_data['time'], unit='ms')
csv_data.to_csv('ozone_data.csv', index=False)

print("Data exported to ozone_data.csv")