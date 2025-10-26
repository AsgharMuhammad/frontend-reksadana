import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Load data
df = pd.read_csv('dataset/Data Historis XKMS.csv')

# Parse volume
def parse_volume(vol_str):
    vol_str = vol_str.replace(',', '.').replace('K', 'e3').replace('M', 'e6')
    try:
        return eval(vol_str)
    except:
        return np.nan

# Preprocessing
df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
df['Volume'] = df['Vol.'].apply(parse_volume)
df['Perubahan'] = df['Perubahan%'].str.replace('%', '', regex=False).str.replace(',', '.').astype(float)
df = df.drop(columns=['Vol.', 'Perubahan%'])

# Urutkan berdasarkan tanggal & tambahkan fitur tambahan
df = df.sort_values('Tanggal')
df['Close_t-1'] = df['Terakhir'].shift(1)
df = df.dropna()

# Hapus outlier ekstrem
df = df[df['Terakhir'] < 10]

# Fitur & target
X = df[['Close_t-1', 'Pembukaan', 'Tertinggi', 'Terendah', 'Volume', 'Perubahan']]
y = df['Terakhir']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Buat dan latih model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Prediksi & evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# Pastikan folder model/ ada
os.makedirs('model', exist_ok=True)

# Simpan model
joblib.dump(model, 'model/random_forest_model.pkl')

print("✅ Model trained & saved successfully!")
