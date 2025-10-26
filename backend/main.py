from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import traceback


app = FastAPI(
    title="API Prediksi Harga Reksa Dana",
    description="Prediksi harga reksa dana berbasis Ensemble Model (RF + GB).",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    models = joblib.load("model_reksadana_rf_final.pkl")
    scaler = joblib.load("scaler_reksadana_rf_final.pkl")

    if isinstance(models, dict):
        rf_model = models['rf_model']
        gb_model = models['gb_model']
        rf_weight = models['rf_weight']
        gb_weight = models['gb_weight']
        print("✅ Ensemble models dan Scaler berhasil dimuat!")
    else:
        rf_model = models
        gb_model = None
        rf_weight = 1.0
        gb_weight = 0.0
        print("✅ Single model dan Scaler berhasil dimuat!")
except Exception as e:
    print("❌ Gagal memuat model atau scaler:", str(e))


def preprocess_data(df: pd.DataFrame) -> tuple:
    try:
        if "Tanggal" not in df.columns:
            raise ValueError("Kolom 'Tanggal' tidak ditemukan dalam dataset.")

        df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
        df = df.sort_values("Tanggal").dropna(subset=["Tanggal"]).reset_index(drop=True)

        if "Terakhir" not in df.columns:
            raise ValueError("Kolom 'Terakhir' (target) tidak ditemukan dalam dataset.")

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

        df['Perubahan'] = df['Terakhir'].pct_change()

        df['Price_Range'] = df['Tertinggi'] - df['Terendah']
        df['Price_Range_Pct'] = (df['Tertinggi'] - df['Terendah']) / df['Terendah']
        df['Price_Momentum'] = df['Terakhir'] - df['Terakhir'].shift(3)

        df['High_Low_Avg'] = (df['Tertinggi'] + df['Terendah']) / 2
        df['Open_Close_Diff'] = df['Pembukaan'] - df['Close_t-1']

        df['High_Close_Diff'] = df['Tertinggi'] - df['Terakhir']
        df['Close_Low_Diff'] = df['Terakhir'] - df['Terendah']

        if 'Volume' in df.columns:
            df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Volume_Ratio'] = df['Terakhir'] / (df['Volume'] + 1)
        else:
            df['Volume'] = 0
            df['Volume_MA_3'] = 0
            df['Volume_MA_5'] = 0
            df['Volume_MA_7'] = 0
            df['Volume_Change'] = 0
            df['Price_Volume_Ratio'] = 0

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

        df = df.dropna().reset_index(drop=True)

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
        y = df["Terakhir"]

        X_scaled = scaler.transform(X)

        return df, X_scaled, y, feature_cols
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Kesalahan preprocessing: {str(e)}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        required_cols = ['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Kolom '{col}' tidak ditemukan di file CSV.")

        df_processed, X_scaled, y_true, feature_cols = preprocess_data(df)

        if gb_model is not None:
            rf_pred = rf_model.predict(X_scaled)
            gb_pred = gb_model.predict(X_scaled)
            y_pred = (rf_weight * rf_pred) + (gb_weight * gb_pred)
        else:
            y_pred = rf_model.predict(X_scaled)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        hasil_df = df_processed[['Tanggal']].copy()
        hasil_df['Actual'] = y_true.values
        hasil_df['Predicted'] = y_pred
        hasil_df['Selisih_(y - y_pred)'] = y_true - y_pred
        hasil_df['Absolut_|y - y_pred|'] = abs(y_true - y_pred)
        hasil_df['Kuadrat_(y - y_pred)^2'] = (y_true - y_pred) ** 2

        return {
            "evaluasi": {
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4)
            },
            "data": hasil_df.to_dict(orient='records')
        }

    except Exception as e:
        print("=== TERJADI ERROR SAAT PREDIKSI ===")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")


@app.get("/")
def home():
    return {"message": "API Prediksi Harga Reksa Dana - Ensemble Model (RF + GB) v4.0 🚀"}
