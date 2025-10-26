# ===============================================================
#  FastAPI Backend untuk Prediksi Harga Reksa Dana
#  Sinkron dengan Model Random Forest (fitur: lag & moving average)
#  Author: Asghar (F1G121065 - Ilmu Komputer UHO)
# ===============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import traceback


# === 1. Inisialisasi Aplikasi FastAPI ===
app = FastAPI(
    title="API Prediksi Harga Reksa Dana",
    description="Prediksi harga reksa dana berbasis Random Forest dengan fitur lag & moving average.",
    version="2.0"
)

# === 2. Konfigurasi CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],  # sesuaikan jika nanti di-deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 3. Load Model dan Scaler ===
try:
    model = joblib.load("model_reksadana_rf_final.pkl")
    scaler = joblib.load("scaler_reksadana_rf_final.pkl")
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print("❌ Gagal memuat model atau scaler:", str(e))


# === 4. Fungsi untuk Feature Engineering ===
def preprocess_data(df: pd.DataFrame) -> tuple:
    try:
        # Pastikan kolom tanggal ada
        if "Tanggal" not in df.columns:
            raise ValueError("Kolom 'Tanggal' tidak ditemukan dalam dataset.")
        
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
        df = df.sort_values("Tanggal").dropna(subset=["Tanggal"]).reset_index(drop=True)

        # Pastikan target ada
        if "Terakhir" not in df.columns:
            raise ValueError("Kolom 'Terakhir' (target) tidak ditemukan dalam dataset.")

        # Buat fitur tambahan
        df["Close_t-1"] = df["Terakhir"].shift(1)
        df['Close_t-2'] = df['Terakhir'].shift(2)
        df["Perubahan"] = df["Terakhir"].pct_change()
        df["MA_3"] = df["Terakhir"].rolling(window=3).mean()  # moving average 3 hari
        df['MA_5'] = df['Terakhir'].rolling(window=5).mean()

        df = df.dropna().reset_index(drop=True)

        # Tentukan fitur yang tersedia (otomatis hanya ambil yang ada)
        fitur_opsional = ["Close_t-1","Close_t-2", "Pembukaan", "Tertinggi", "Terendah", "Volume", "Perubahan", "MA_3", "MA_5"]
        fitur_ada = [col for col in fitur_opsional if col in df.columns]

        if len(fitur_ada) == 0:
            raise ValueError("Tidak ada kolom fitur yang sesuai ditemukan dalam dataset.")

        # Ambil fitur & target
        X = df[fitur_ada]
        y = df["Terakhir"]

        # Transformasi (pastikan scaler sesuai)
        X_scaled = scaler.transform(X)

        return df, X_scaled, y, fitur_ada
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Kesalahan preprocessing: {str(e)}")

# === 5. Endpoint Utama ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # === Baca CSV ===
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # === Pastikan kolom wajib tersedia ===
        required_cols = ['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Kolom '{col}' tidak ditemukan di file CSV.")

        # === Preprocessing data (fitur turunan termasuk Moving Average) ===
        df_processed, X_scaled, y_true, fitur_ada = preprocess_data(df)

        # === Tentukan fitur yang digunakan sesuai model training ===
        fitur = ['Close_t-1', 'Close_t-2', 'Pembukaan', 'Tertinggi', 'Terendah', 'Perubahan', 'MA_3', 'MA_5']

        # Pastikan hanya fitur yang tersedia yang digunakan
        fitur_ada = [f for f in fitur if f in df_processed.columns]

        # Gunakan DataFrame hasil preprocessing
        X = df_processed[fitur_ada]
        y_true = df_processed['Terakhir']

        # === Normalisasi data ===
        X_scaled = scaler.transform(X)

        # === Prediksi ===
        y_pred = model.predict(X_scaled)

        # === Evaluasi ===
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # === Buat DataFrame hasil ===
        hasil_df = df_processed[['Tanggal']].copy()
        hasil_df['Actual'] = y_true.values
        hasil_df['Predicted'] = y_pred
        hasil_df['Selisih_(y - y_pred)'] = y_true - y_pred
        hasil_df['Absolut_|y - y_pred|'] = abs(y_true - y_pred)
        hasil_df['Kuadrat_(y - y_pred)^2'] = (y_true - y_pred) ** 2

        # === Kembalikan ke frontend ===
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



# === 6. Endpoint Root ===
@app.get("/")
def home():
    return {"message": "API Prediksi Harga Reksa Dana - versi final dengan Moving Average & Lag Features 🚀"}
