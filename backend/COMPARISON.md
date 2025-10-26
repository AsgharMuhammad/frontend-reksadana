# Before vs After Comparison

## Feature Count
- **Before**: 6 features
- **After**: 22 features

## Detailed Feature Comparison

### Original Features (6)
1. Close_t-1 (lag 1)
2. Pembukaan (Opening price)
3. Tertinggi (High price)
4. Terendah (Low price)
5. Volume
6. Perubahan (Price change %)

### New Enhanced Features (22)

#### Lag Features (3)
1. Close_t-1
2. Close_t-2 ✨ NEW
3. Close_t-3 ✨ NEW

#### Price Features (3)
4. Pembukaan
5. Tertinggi
6. Terendah

#### Volume & Change (2)
7. Volume
8. Perubahan (now calculated as pct_change)

#### Moving Averages (3) ✨ NEW
9. MA_3 (3-day moving average)
10. MA_5 (5-day moving average)
11. MA_7 (7-day moving average)

#### Exponential Moving Averages (2) ✨ NEW
12. EMA_3 (3-day exponential MA)
13. EMA_5 (5-day exponential MA)

#### Volatility Features (2) ✨ NEW
14. Volatility_5 (5-day rolling std)
15. Volatility_10 (10-day rolling std)

#### Price Range Features (3) ✨ NEW
16. Price_Range (High - Low)
17. Price_Range_Pct (percentage range)
18. High_Low_Avg (average of high/low)

#### Derived Features (2) ✨ NEW
19. Open_Close_Diff (gap detection)

#### Volume Features (2) ✨ NEW
20. Volume_MA_3 (3-day volume MA)
21. Volume_MA_5 (5-day volume MA)

#### Technical Indicators (1) ✨ NEW
22. RSI (Relative Strength Index)

## Model Configuration

### Before
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```
- No feature scaling
- Fixed hyperparameters
- No cross-validation

### After
```python
GridSearchCV with:
  n_estimators: [200, 300]
  max_depth: [15, 20]
  min_samples_split: [2, 5]
  min_samples_leaf: [1, 2]
  max_features: ['sqrt', 'log2']
  cv=5
```
- StandardScaler normalization ✨
- Hyperparameter optimization ✨
- 5-fold cross-validation ✨
- Parallel processing (n_jobs=-1) ✨

## Training Process

### Before
1. Load data
2. Basic preprocessing
3. Split train/test
4. Train with fixed params
5. Evaluate

### After
1. Load data
2. Advanced feature engineering (22 features)
3. Feature scaling with StandardScaler ✨
4. Split train/test
5. GridSearch with 5-fold CV ✨
6. Train best model
7. Display feature importances ✨
8. Evaluate with multiple metrics (MSE, RMSE, MAE, R²)
9. Save both model and scaler ✨

## Key Improvements

### 1. Feature Engineering
- 16 additional engineered features
- Captures temporal patterns (lags, MAs)
- Market dynamics (volatility, RSI)
- Price relationships (ranges, gaps)

### 2. Feature Scaling
- All features normalized to same scale
- Prevents dominance by large-valued features
- Improves model convergence

### 3. Hyperparameter Optimization
- Automated search for best parameters
- Cross-validation for robust evaluation
- Prevents overfitting

### 4. Model Complexity
- More trees (200-300 vs 100)
- Deeper trees (15-20 vs 10)
- Better regularization parameters

## Expected Impact on Metrics

### MSE (Mean Squared Error)
- Lower value = better fit
- Squared errors penalize large deviations
- **Expected**: Significant reduction

### RMSE (Root Mean Squared Error)
- Same unit as target variable
- More interpretable than MSE
- **Expected**: Reduced by 30-50%

### MAE (Mean Absolute Error)
- Average absolute prediction error
- Less sensitive to outliers than MSE
- **Expected**: Reduced by 25-40%

### R² (Coefficient of Determination)
- Proportion of variance explained
- Range: 0 to 1 (higher is better)
- **Expected**: Increased significantly

## Why These Changes Work

1. **More Information**: 22 features vs 6 gives model more context
2. **Better Patterns**: Technical indicators capture market behavior
3. **Proper Scaling**: Prevents bias toward large-valued features
4. **Optimal Parameters**: GridSearch finds best model configuration
5. **Validation**: Cross-validation ensures generalization
6. **Complexity**: Deeper, larger forest can learn complex patterns

## To Apply Changes

Simply run the new training script:
```bash
cd backend
python3 train_model.py
```

The script will automatically:
- Generate all 22 features
- Optimize hyperparameters
- Train the best model
- Save model and scaler
- Display improved metrics
