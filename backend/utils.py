import numpy as np
import pandas as pd

def parse_volume(vol_str):
    vol_str = vol_str.replace(',', '.').replace('K', 'e3').replace('M', 'e6')
    try:
        return eval(vol_str)
    except:
        return np.nan

def preprocess_data(df):
    # Sama seperti train_model.py
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df['Volume'] = df['Vol.'].apply(parse_volume)
    df['Perubahan'] = df['Perubahan%'].str.replace('%', '').str.replace(',', '.').astype(float)
    df = df.drop(columns=['Vol.', 'Perubahan%'])
    df = df.sort_values('Tanggal')
    df['Close_t-1'] = df['Terakhir'].shift(1)
    df = df.dropna()
    df = df[df['Terakhir'] < 10]

    # Fitur dan target
    X = df[['Close_t-1', 'Pembukaan', 'Tertinggi', 'Terendah', 'Volume', 'Perubahan']]
    y = df['Terakhir']

    return X, y
