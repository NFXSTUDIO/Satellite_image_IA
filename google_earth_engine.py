import ee
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input # Import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Initialize Earth Engine
ee.Initialize(project='ee-arthurcourbevoie')

# 1. Data Retrieval from Google Earth Engine

# Define the dataset and time range
dataset = ee.ImageCollection('TOMS/MERGED')
start_date = '1980-01-01'
end_date = '2024-12-31'

# Filter the dataset
filtered_dataset = dataset.filterDate(start_date, end_date)

# Define a region of interest (e.g., a bounding box over Europe)
roi = ee.Geometry.Rectangle([[-75, 59], [-20, 84]])

# Function to extract ozone values
def extract_ozone(image):
    ozone_image = image.select('ozone')
    mean_ozone = ozone_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10000  # Adjust scale as needed
    ).get('ozone')
    return image.set('mean_ozone', mean_ozone)

# Apply the function to the image collection
ozone_collection = filtered_dataset.map(extract_ozone)

# Get the list of mean ozone values and dates
data_list = ozone_collection.reduceColumns(ee.Reducer.toList(2), ['system:time_start', 'mean_ozone']).values().get(0).getInfo()

# Convert the data to a pandas DataFrame
data = pd.DataFrame(data_list, columns=['time', 'ozone'])
data['time'] = pd.to_datetime(data['time'], unit='ms')

# Clean the data, removing NaN values
data = data.dropna()

# 2. Data Preprocessing

# Feature Engineering (example: extract day of year)
data['day_of_year'] = data['time'].dt.dayofyear

# Feature Scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[['ozone', 'day_of_year']])
scaled_data = pd.DataFrame(scaled_features, columns=['ozone_scaled', 'day_of_year_scaled'])

# Prepare data for machine learning
X = scaled_data[['day_of_year_scaled']].values
print(X)
y = scaled_data['ozone_scaled'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),]

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256,activation='relu'),
    BatchNormalization(),
    Dropout(0.8),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.0001), metrics=[rmse,'accuracy'], loss='mean_squared_error') # Corrected argument name
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test,y_test), callbacks=callbacks)
model.save('Ozone_model.h5')

# Make predictions on the test set
y_pred = model.predict(X_test)

t = list(range(1, len(y_test) + 1))
#Plotting the results
plt.figure(figsize=(15, 6))
plt.plot(t, y_test, label='Actual Ozone Concentration')
plt.plot(t, y_pred, label='Predicted Ozone Concentration')
plt.xlabel('Time Step')
plt.ylabel('Scaled Ozone Concentration')
plt.title('Actual vs. Predicted Scaled Ozone Concentration')
plt.legend()
plt.grid(True)
plt.show()

dates = [pd.to_datetime(data[0], unit='ms') for data in data_list]
ozone_values = [data[1] for data in data_list]
df = pd.DataFrame({'date': dates, 'ozone': ozone_values})

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['ozone'],label= 'Actual Ozone Concentration')
plt.xlabel('Date')
plt.ylabel('Mean Ozone (mol/m²)')
plt.title('Mean Ozone Concentration over Europe')
plt.grid(True)
plt.show()

# Exporting the extracted data to CSV
csv_data = pd.DataFrame(data_list, columns=['time', 'ozone'])
csv_data['time'] = pd.to_datetime(csv_data['time'], unit='ms')
csv_data.to_csv('ozone_data.csv', index=False)

print("Data exported to ozone_data.csv")