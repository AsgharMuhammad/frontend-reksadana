import numpy as np
import pandas as pd

def parse_volume(vol_str):
    vol_str = vol_str.replace(',', '.').replace('K', 'e3').replace('M', 'e6')
    try:
        return eval(vol_str)
    except:
        return np.nan

def engineer_features(df):
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

    return df

def preprocess_data(df):
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Volume'] = df['Vol.'].apply(parse_volume)
    df['Perubahan'] = df['Perubahan%'].str.replace('%', '', regex=False).str.replace(',', '.').astype(float)
    df = df.drop(columns=['Vol.', 'Perubahan%'])
    df = df.sort_values('Tanggal').reset_index(drop=True)

    df = engineer_features(df)

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

    return X, y
