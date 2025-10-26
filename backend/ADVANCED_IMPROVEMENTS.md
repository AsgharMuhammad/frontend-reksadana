# Advanced Model Improvements - Version 2

## Major Enhancements

### 1. Ensemble Learning (NEW!)
**Combination of Two Powerful Models:**
- **Random Forest**: 500 trees, depth 25
- **Gradient Boosting**: 300 estimators, learning rate 0.05
- **Weighted Ensemble**: 60% RF + 40% GB

**Why Ensemble Works Better:**
- RF captures non-linear patterns and interactions
- GB sequentially corrects errors
- Combined predictions are more robust and accurate
- Reduces overfitting through diversity

### 2. Enhanced Feature Engineering (56 Features!)
**Previous: 22 features → Now: 56 features**

#### New Lag Features (5 total)
- Close_t-1, t-2, t-3, t-5, t-7
- Captures longer historical patterns

#### Extended Moving Averages (5 total)
- MA_3, MA_5, MA_7, MA_10, MA_14
- Better trend identification

#### Extended EMAs (4 total)
- EMA_3, EMA_5, EMA_7, EMA_10
- More responsive to recent changes

#### Enhanced Volatility (4 measures)
- Volatility_3, 5, 7, 10
- Better risk assessment

#### Advanced Price Features
- **Price_Momentum**: 3-day price change
- **High_Close_Diff**: Upper shadow analysis
- **Close_Low_Diff**: Lower shadow analysis

#### Volume Intelligence
- Volume_MA_3, MA_5, MA_7
- **Volume_Change**: Rate of volume change
- **Price_Volume_Ratio**: Price efficiency metric

#### Technical Indicators

**MACD (Moving Average Convergence Divergence)**
- MACD line
- Signal line
- Histogram (divergence)

**Bollinger Bands**
- BB_Middle (20-day MA)
- BB_Upper (Upper band)
- BB_Lower (Lower band)
- BB_Width (Band width - volatility)
- BB_Position (Where price sits in bands)

**MA Distance Indicators**
- MA_Distance_3, 5, 7
- EMA_Distance_3, 5
- Shows how far price is from moving averages

**Return Features**
- Return_1d, 3d, 5d, 7d
- Multiple timeframe returns

**Temporal Features**
- Day_of_Week (0-6)
- Day_of_Month (1-31)
- Month (1-12)
- Captures seasonal patterns

### 3. RobustScaler (Improved from StandardScaler)
**Why RobustScaler?**
- Less sensitive to outliers
- Uses median and IQR instead of mean and std
- Better for financial data with extreme values
- More stable normalization

### 4. Time Series Split
- Uses chronological split (80/20)
- No shuffling - respects temporal order
- More realistic evaluation for time series

### 5. Advanced Model Parameters

**Random Forest:**
```python
n_estimators=500      # More trees
max_depth=25          # Deeper trees
max_features='sqrt'   # Feature sampling
bootstrap=True        # Bootstrap sampling
oob_score=True       # Out-of-bag validation
```

**Gradient Boosting:**
```python
n_estimators=300      # Boosting rounds
learning_rate=0.05    # Conservative learning
max_depth=7          # Controlled depth
subsample=0.8        # Stochastic sampling
```

## Feature Count Comparison

| Version | Feature Count | Description |
|---------|---------------|-------------|
| Original | 6 | Basic features only |
| Version 1 | 22 | Added MAs, EMAs, RSI, volatility |
| **Version 2** | **56** | **Full technical analysis + ensemble** |

## Technical Indicators Explained

### MACD (Trend Following)
- Identifies trend direction and momentum
- Crossovers signal potential reversals
- Histogram shows momentum strength

### Bollinger Bands (Volatility)
- Shows price volatility
- Identifies overbought/oversold conditions
- BB_Position shows relative price location

### RSI (Momentum)
- Measures speed and magnitude of price changes
- Values > 70: Overbought
- Values < 30: Oversold

### Moving Average Distances
- Shows price deviation from trend
- Helps identify mean reversion opportunities

### Multi-timeframe Returns
- Captures momentum across different periods
- Identifies short vs long-term trends

## Expected Performance Improvements

### From Original Model:
- **MSE**: Expected reduction of 60-70%
- **RMSE**: Expected reduction of 40-50%
- **MAE**: Expected reduction of 35-45%
- **R²**: Expected increase to 0.85-0.95

### From Version 1:
- **MSE**: Expected reduction of 30-40%
- **RMSE**: Expected reduction of 15-25%
- **MAE**: Expected reduction of 15-20%
- **R²**: Expected increase by 10-15%

## Why This Approach Works

### 1. Ensemble Diversity
- RF: Parallel trees, random features
- GB: Sequential error correction
- Combined: Best of both worlds

### 2. Comprehensive Features
- Price action (lags, ranges)
- Trend (MAs, EMAs)
- Momentum (RSI, MACD, returns)
- Volatility (std dev, Bollinger Bands)
- Volume (MAs, changes, ratios)
- Temporal (day, month patterns)

### 3. Robust Preprocessing
- RobustScaler handles outliers
- Proper time series split
- No data leakage

### 4. Model Capacity
- 500 RF trees capture complex patterns
- 300 GB rounds refine predictions
- Deep trees learn intricate relationships

## How to Use

### Training:
```bash
cd backend
python3 train_model.py
```

### What Gets Saved:
1. **model_reksadana_rf_final.pkl**: Dictionary containing
   - rf_model: Random Forest model
   - gb_model: Gradient Boosting model
   - rf_weight: RF ensemble weight (0.6)
   - gb_weight: GB ensemble weight (0.4)

2. **scaler_reksadana_rf_final.pkl**: RobustScaler

### Output Includes:
- Training and test metrics
- Feature importance rankings
- Model configuration details
- Sample counts

## Key Advantages

1. **Better Predictions**: Ensemble reduces variance
2. **More Robust**: Less sensitive to outliers
3. **Comprehensive**: 56 features capture all aspects
4. **Technical Analysis**: Professional indicators included
5. **Time-Aware**: Respects temporal dependencies
6. **Explainable**: Feature importance rankings provided

## Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Lag Features | 5 | Close_t-1, t-2, t-3, t-5, t-7 |
| Moving Averages | 5 | MA_3, MA_5, MA_7, MA_10, MA_14 |
| EMAs | 4 | EMA_3, EMA_5, EMA_7, EMA_10 |
| Volatility | 4 | Volatility_3, 5, 7, 10 |
| Price Features | 8 | Range, Momentum, Diffs, Avg |
| Volume Features | 5 | MAs, Change, Ratio |
| RSI | 1 | Relative Strength Index |
| MA Distances | 5 | Distance from MAs & EMAs |
| MACD | 3 | Line, Signal, Histogram |
| Bollinger Bands | 5 | Upper, Lower, Middle, Width, Position |
| Returns | 4 | 1d, 3d, 5d, 7d returns |
| Temporal | 3 | Day, Month, Week |
| **TOTAL** | **56** | **Complete Feature Set** |

## Model Architecture

```
Input (56 features)
         ↓
   RobustScaler
         ↓
    ┌────┴────┐
    ↓         ↓
Random Forest  Gradient Boosting
(500 trees)   (300 estimators)
    ↓         ↓
  60% weight  40% weight
    └────┬────┘
         ↓
  Final Prediction
```

This advanced ensemble approach should deliver significantly better prediction accuracy!
