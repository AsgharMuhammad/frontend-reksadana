import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
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

df = df.sort_values('Tanggal').reset_index(drop=True)

df['Close_t-1'] = df['Terakhir'].shift(1)
df['Close_t-2'] = df['Terakhir'].shift(2)
df['Close_t-3'] = df['Terakhir'].shift(3)
df['Close_t-5'] = df['Terakhir'].shift(5)
df['Close_t-7'] = df['Terakhir'].shift(7)

df['MA_3'] = df['Terakhir'].rolling(window=3).mean()
df['MA_5'] = df['Terakhir'].rolling(window=5).mean()
df['MA_7'] = df['Terakhir'].rolling(window=7).mean()
df['MA_10'] = df['Terakhir'].rolling(window=10).mean()
df['MA_14'] = df['Terakhir'].rolling(window=14).mean()

df['EMA_3'] = df['Terakhir'].ewm(span=3, adjust=False).mean()
df['EMA_5'] = df['Terakhir'].ewm(span=5, adjust=False).mean()
df['EMA_7'] = df['Terakhir'].ewm(span=7, adjust=False).mean()
df['EMA_10'] = df['Terakhir'].ewm(span=10, adjust=False).mean()

df['Volatility_3'] = df['Terakhir'].rolling(window=3).std()
df['Volatility_5'] = df['Terakhir'].rolling(window=5).std()
df['Volatility_7'] = df['Terakhir'].rolling(window=7).std()
df['Volatility_10'] = df['Terakhir'].rolling(window=10).std()

df['Price_Range'] = df['Tertinggi'] - df['Terendah']
df['Price_Range_Pct'] = (df['Tertinggi'] - df['Terendah']) / df['Terendah']
df['Price_Momentum'] = df['Terakhir'] - df['Terakhir'].shift(3)

df['High_Low_Avg'] = (df['Tertinggi'] + df['Terendah']) / 2
df['Open_Close_Diff'] = df['Pembukaan'] - df['Close_t-1']

df['High_Close_Diff'] = df['Tertinggi'] - df['Terakhir']
df['Close_Low_Diff'] = df['Terakhir'] - df['Terendah']

df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
df['Volume_Change'] = df['Volume'].pct_change()

df['Price_Volume_Ratio'] = df['Terakhir'] / (df['Volume'] + 1)

gains = df['Terakhir'].diff().clip(lower=0)
losses = -df['Terakhir'].diff().clip(upper=0)
avg_gain = gains.rolling(window=14).mean()
avg_loss = losses.rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-10)
df['RSI'] = 100 - (100 / (1 + rs))

df['MA_Distance_3'] = (df['Terakhir'] - df['MA_3']) / df['MA_3']
df['MA_Distance_5'] = (df['Terakhir'] - df['MA_5']) / df['MA_5']
df['MA_Distance_7'] = (df['Terakhir'] - df['MA_7']) / df['MA_7']

df['EMA_Distance_3'] = (df['Terakhir'] - df['EMA_3']) / df['EMA_3']
df['EMA_Distance_5'] = (df['Terakhir'] - df['EMA_5']) / df['EMA_5']

exp12 = df['Terakhir'].ewm(span=12, adjust=False).mean()
exp26 = df['Terakhir'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp12 - exp26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

bb_window = 20
df['BB_Middle'] = df['Terakhir'].rolling(window=bb_window).mean()
bb_std = df['Terakhir'].rolling(window=bb_window).std()
df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
df['BB_Position'] = (df['Terakhir'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

df['Return_1d'] = df['Terakhir'].pct_change(1)
df['Return_3d'] = df['Terakhir'].pct_change(3)
df['Return_5d'] = df['Terakhir'].pct_change(5)
df['Return_7d'] = df['Terakhir'].pct_change(7)

df['Day_of_Week'] = df['Tanggal'].dt.dayofweek
df['Day_of_Month'] = df['Tanggal'].dt.day
df['Month'] = df['Tanggal'].dt.month

df = df.dropna()
df = df[df['Terakhir'] < 10]

feature_cols = [
    'Close_t-1', 'Close_t-2', 'Close_t-3', 'Close_t-5', 'Close_t-7',
    'Pembukaan', 'Tertinggi', 'Terendah',
    'Volume', 'Perubahan',
    'MA_3', 'MA_5', 'MA_7', 'MA_10', 'MA_14',
    'EMA_3', 'EMA_5', 'EMA_7', 'EMA_10',
    'Volatility_3', 'Volatility_5', 'Volatility_7', 'Volatility_10',
    'Price_Range', 'Price_Range_Pct', 'Price_Momentum',
    'High_Low_Avg', 'Open_Close_Diff',
    'High_Close_Diff', 'Close_Low_Diff',
    'Volume_MA_3', 'Volume_MA_5', 'Volume_MA_7', 'Volume_Change',
    'Price_Volume_Ratio',
    'RSI',
    'MA_Distance_3', 'MA_Distance_5', 'MA_Distance_7',
    'EMA_Distance_3', 'EMA_Distance_5',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
    'Return_1d', 'Return_3d', 'Return_5d', 'Return_7d',
    'Day_of_Week', 'Day_of_Month', 'Month'
]

X = df[feature_cols]
y = df['Terakhir']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(len(X_scaled) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("Training Advanced Random Forest Model...")
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)

print("\nTraining Gradient Boosting Model...")
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gb_model.fit(X_train, y_train)

print("\nCreating Ensemble Model...")
rf_pred_train = rf_model.predict(X_train)
gb_pred_train = gb_model.predict(X_train)
rf_weight = 0.6
gb_weight = 0.4

y_pred_train = (rf_weight * rf_pred_train) + (gb_weight * gb_pred_train)

rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
y_pred = (rf_weight * rf_pred) + (gb_weight * gb_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"ENSEMBLE MODEL EVALUATION METRICS")
print(f"{'='*50}")
print(f"MSE  : {mse:.8f}")
print(f"RMSE : {rmse:.8f}")
print(f"MAE  : {mae:.8f}")
print(f"R²   : {r2:.8f}")
print(f"{'='*50}")

feature_importance_rf = pd.DataFrame({
    'feature': feature_cols,
    'importance_rf': rf_model.feature_importances_
})

feature_importance_gb = pd.DataFrame({
    'feature': feature_cols,
    'importance_gb': gb_model.feature_importances_
})

feature_importance = feature_importance_rf.merge(feature_importance_gb, on='feature')
feature_importance['importance_avg'] = (feature_importance['importance_rf'] + feature_importance['importance_gb']) / 2
feature_importance = feature_importance.sort_values('importance_avg', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance[['feature', 'importance_avg']].head(15).to_string(index=False))

os.makedirs('model', exist_ok=True)

ensemble_models = {
    'rf_model': rf_model,
    'gb_model': gb_model,
    'rf_weight': rf_weight,
    'gb_weight': gb_weight
}

joblib.dump(ensemble_models, 'model_reksadana_rf_final.pkl')
joblib.dump(scaler, 'scaler_reksadana_rf_final.pkl')

print(f"\n{'='*50}")
print("✅ Ensemble model and scaler saved successfully!")
print(f"{'='*50}")
print(f"\nModel Details:")
print(f"- Random Forest: {rf_model.n_estimators} trees, depth {rf_model.max_depth}")
print(f"- Gradient Boosting: {gb_model.n_estimators} estimators, LR {gb_model.learning_rate}")
print(f"- Ensemble Weights: RF={rf_weight}, GB={gb_weight}")
print(f"- Features: {len(feature_cols)} engineered features")
print(f"- Training samples: {len(X_train)}, Test samples: {len(X_test)}")
