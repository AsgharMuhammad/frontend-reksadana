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
    description="Prediksi harga reksa dana berbasis Random Forest dengan fitur lag & moving average.",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("model_reksadana_rf_final.pkl")
    scaler = joblib.load("scaler_reksadana_rf_final.pkl")
    print("✅ Model dan Scaler berhasil dimuat!")
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

        df['MA_3'] = df['Terakhir'].rolling(window=3).mean()
        df['MA_5'] = df['Terakhir'].rolling(window=5).mean()
        df['MA_7'] = df['Terakhir'].rolling(window=7).mean()

        df['EMA_3'] = df['Terakhir'].ewm(span=3, adjust=False).mean()
        df['EMA_5'] = df['Terakhir'].ewm(span=5, adjust=False).mean()

        df['Volatility_5'] = df['Terakhir'].rolling(window=5).std()
        df['Volatility_10'] = df['Terakhir'].rolling(window=10).std()

        df['Perubahan'] = df['Terakhir'].pct_change()

        df['Price_Range'] = df['Tertinggi'] - df['Terendah']
        df['Price_Range_Pct'] = (df['Tertinggi'] - df['Terendah']) / df['Terendah']

        df['High_Low_Avg'] = (df['Tertinggi'] + df['Terendah']) / 2
        df['Open_Close_Diff'] = df['Pembukaan'] - df['Close_t-1']

        if 'Volume' in df.columns:
            df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        else:
            df['Volume'] = 0
            df['Volume_MA_3'] = 0
            df['Volume_MA_5'] = 0

        df['RSI'] = 100 - (100 / (1 + (df['Terakhir'].diff().clip(lower=0).rolling(14).mean() /
                                      (-df['Terakhir'].diff().clip(upper=0).rolling(14).mean()))))

        df = df.dropna().reset_index(drop=True)

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

        y_pred = model.predict(X_scaled)

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
    return {"message": "API Prediksi Harga Reksa Dana - Enhanced version with advanced features 🚀"}
