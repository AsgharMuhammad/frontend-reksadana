import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

df = pd.read_csv('dataset/Data Historis XKMS.csv')

def parse_volume(vol_str):
    vol_str = vol_str.replace(',', '.').replace('K', 'e3').replace('M', 'e6')
    try:
        return eval(vol_str)
    except:
        return np.nan

df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
df['Volume'] = df['Vol.'].apply(parse_volume)
df['Perubahan'] = df['Perubahan%'].str.replace('%', '', regex=False).str.replace(',', '.').astype(float)
df = df.drop(columns=['Vol.', 'Perubahan%'])

df = df.sort_values('Tanggal')

df['Close_t-1'] = df['Terakhir'].shift(1)
df['Close_t-2'] = df['Terakhir'].shift(2)
df['Close_t-3'] = df['Terakhir'].shift(3)

df['MA_3'] = df['Terakhir'].rolling(window=3).mean()
df['MA_5'] = df['Terakhir'].rolling(window=5).mean()
df['MA_7'] = df['Terakhir'].rolling(window=7).mean()

df['EMA_3'] = df['Terakhir'].ewm(span=3, adjust=False).mean()
df['EMA_5'] = df['Terakhir'].ewm(span=5, adjust=False).mean()

df['Volatility_5'] = df['Terakhir'].rolling(window=5).std()
df['Volatility_10'] = df['Terakhir'].rolling(window=10).std()

df['Price_Range'] = df['Tertinggi'] - df['Terendah']
df['Price_Range_Pct'] = (df['Tertinggi'] - df['Terendah']) / df['Terendah']

df['High_Low_Avg'] = (df['Tertinggi'] + df['Terendah']) / 2
df['Open_Close_Diff'] = df['Pembukaan'] - df['Close_t-1']

df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

df['RSI'] = 100 - (100 / (1 + (df['Terakhir'].diff().clip(lower=0).rolling(14).mean() /
                              (-df['Terakhir'].diff().clip(upper=0).rolling(14).mean()))))

df = df.dropna()

df = df[df['Terakhir'] < 10]

feature_cols = [
    'Close_t-1', 'Close_t-2', 'Close_t-3',
    'Pembukaan', 'Tertinggi', 'Terendah',
    'Volume', 'Perubahan',
    'MA_3', 'MA_5', 'MA_7',
    'EMA_3', 'EMA_5',
    'Volatility_5', 'Volatility_10',
    'Price_Range', 'Price_Range_Pct',
    'High_Low_Avg', 'Open_Close_Diff',
    'Volume_MA_3', 'Volume_MA_5',
    'RSI'
]

X = df[feature_cols]
y = df['Terakhir']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print("Training model with GridSearchCV...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Evaluation Metrics ===")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== Top 10 Feature Importances ===")
print(feature_importance.head(10))

os.makedirs('model', exist_ok=True)

joblib.dump(best_model, 'model_reksadana_rf_final.pkl')
joblib.dump(scaler, 'scaler_reksadana_rf_final.pkl')

print("\n✅ Model and scaler saved successfully!")
