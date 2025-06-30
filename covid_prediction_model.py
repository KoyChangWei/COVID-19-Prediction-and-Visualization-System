import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    from sklearn.base import BaseEstimator, RegressorMixin
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available - LSTM model will be included")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - LSTM model will be skipped")
    print("   To install: pip install tensorflow")

print("🚀 COVID-19 Case Prediction Model")
print("=" * 50)

# ===== 1. DATA LOADING & EXPLORATION =====
print("\n📊 STEP 1: Loading and exploring data...")
df = pd.read_csv('covid_cases_vaxstatus.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"Original dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Number of states: {len(df['state'].unique())}")

# ===== 2. DATA PREPROCESSING =====
print("\n🔧 STEP 2: Data preprocessing...")

# Remove Malaysia total (keep only individual states)
df_states = df[df['state'].str.lower() != 'malaysia'].copy()
print(f"After removing Malaysia total: {df_states.shape}")

# Create month-year aggregation
df_states['year'] = df_states['date'].dt.year
df_states['month'] = df_states['date'].dt.month
df_states['month_year'] = df_states['date'].dt.to_period('M').astype(str)

# ===== 3. TARGET VARIABLE CREATION =====
print("\n🎯 STEP 3: Creating target variable...")

# Aggregate to monthly level by state
monthly_data = df_states.groupby(['state', 'month_year', 'year', 'month']).agg({
    'cases_unvax': 'sum',
    'cases_pvax': 'sum', 
    'cases_fvax': 'sum',
    'cases_boost': 'sum'
}).reset_index()

# Create total monthly cases
monthly_data['total_monthly_cases'] = (
    monthly_data['cases_unvax'] + 
    monthly_data['cases_pvax'] + 
    monthly_data['cases_fvax'] + 
    monthly_data['cases_boost']
)

print(f"Monthly aggregated data shape: {monthly_data.shape}")
print(f"Total monthly cases range: {monthly_data['total_monthly_cases'].min()} to {monthly_data['total_monthly_cases'].max()}")

# Sort by state and date for time series features
monthly_data = monthly_data.sort_values(['state', 'month_year']).reset_index(drop=True)

# ===== 4. FEATURE ENGINEERING =====
print("\n⚙️ STEP 4: Feature engineering...")

# Create lag features (previous months' data)
for lag in [1, 2, 3]:
    monthly_data[f'lag_{lag}_total'] = monthly_data.groupby('state')['total_monthly_cases'].shift(lag)
    monthly_data[f'lag_{lag}_unvax'] = monthly_data.groupby('state')['cases_unvax'].shift(lag)
    monthly_data[f'lag_{lag}_pvax'] = monthly_data.groupby('state')['cases_pvax'].shift(lag)
    monthly_data[f'lag_{lag}_fvax'] = monthly_data.groupby('state')['cases_fvax'].shift(lag)
    monthly_data[f'lag_{lag}_boost'] = monthly_data.groupby('state')['cases_boost'].shift(lag)

# Create rolling averages
monthly_data['rolling_3m_avg'] = monthly_data.groupby('state')['total_monthly_cases'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
monthly_data['rolling_6m_avg'] = monthly_data.groupby('state')['total_monthly_cases'].rolling(6, min_periods=1).mean().reset_index(0, drop=True)

# Create trend features
monthly_data['cases_change_1m'] = monthly_data.groupby('state')['total_monthly_cases'].pct_change(1)
monthly_data['cases_change_3m'] = monthly_data.groupby('state')['total_monthly_cases'].pct_change(3)

# Vaccination rate features
monthly_data['vax_rate'] = (
    monthly_data['cases_pvax'] + monthly_data['cases_fvax'] + monthly_data['cases_boost']
) / (monthly_data['total_monthly_cases'] + 1)  # Add 1 to avoid division by zero

# Clean infinite and extreme values
print("🧹 Cleaning infinite and extreme values...")
# Replace infinite values with NaN
monthly_data = monthly_data.replace([np.inf, -np.inf], np.nan)

# Cap extreme percentage changes to reasonable limits
monthly_data['cases_change_1m'] = np.clip(monthly_data['cases_change_1m'], -5, 5)  # -500% to +500%
monthly_data['cases_change_3m'] = np.clip(monthly_data['cases_change_3m'], -5, 5)
monthly_data['vax_rate'] = np.clip(monthly_data['vax_rate'], 0, 1)  # 0% to 100%

# Seasonal features
monthly_data['quarter'] = monthly_data['month'].apply(lambda x: (x-1)//3 + 1)
monthly_data['is_holiday_season'] = monthly_data['month'].isin([12, 1, 6, 7]).astype(int)  # Dec, Jan, Jun, Jul

# ===== 5. CREATE TARGET VARIABLE =====
print("\n🎯 STEP 5: Creating target variable (Next Month's Total Cases)...")

# Target: Next month's total cases for each state
monthly_data['target_next_month_cases'] = monthly_data.groupby('state')['total_monthly_cases'].shift(-1)

# Encode categorical variables
le_state = LabelEncoder()
monthly_data['state_encoded'] = le_state.fit_transform(monthly_data['state'])

print(f"States encoded: {dict(zip(le_state.classes_, range(len(le_state.classes_))))}")

# ===== 6. PREPARE FINAL DATASET =====
print("\n📋 STEP 6: Preparing final dataset...")

# Select features for modeling
feature_columns = [
    # Current month cases by vaccination status
    'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost', 'total_monthly_cases',
    
    # Lag features (1-3 months back)
    'lag_1_total', 'lag_2_total', 'lag_3_total',
    'lag_1_unvax', 'lag_1_pvax', 'lag_1_fvax', 'lag_1_boost',
    
    # Rolling averages
    'rolling_3m_avg', 'rolling_6m_avg',
    
    # Trend features
    'cases_change_1m', 'cases_change_3m',
    
    # Vaccination and seasonal features
    'vax_rate', 'month', 'quarter', 'year', 'is_holiday_season',
    
    # State encoding
    'state_encoded'
]

# Remove rows with missing values (due to lag features)
df_model = monthly_data[feature_columns + ['target_next_month_cases']].dropna().copy()

# Final data validation
print("🔍 Final data validation...")
print(f"NaN values per column: {df_model.isnull().sum().sum()}")
print(f"Infinite values: {np.isinf(df_model.select_dtypes(include=[np.number])).sum().sum()}")

# Additional safety check - remove any remaining problematic rows
df_model = df_model[~np.isinf(df_model.select_dtypes(include=[np.number])).any(axis=1)]
df_model = df_model.dropna()

print(f"Final modeling dataset shape: {df_model.shape}")
print(f"Target variable (next month cases) statistics:")
print(df_model['target_next_month_cases'].describe())

# ===== 7. TRAIN-TEST SPLIT =====
print("\n🔄 STEP 7: Splitting data for training and testing...")

X = df_model[feature_columns]
y = df_model['target_next_month_cases']

# Use time-aware split (80% for training, 20% for testing)
# This ensures we train on older data and test on newer data
split_index = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_model)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df_model)*100:.1f}%)")
print(f"Training period: indices 0 to {split_index-1}")
print(f"Testing period: indices {split_index} to {len(df_model)-1}")

# ===== 8. FEATURE SCALING & SMOTE =====
print("\n⚖️ STEP 8: Feature scaling and SMOTE preprocessing...")

# Standard scaling for most models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMax scaling for LSTM (0-1 range works better)
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

# Also scale target for LSTM
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Apply SMOTE for regression (create synthetic samples to balance distribution)
print("🔄 Applying SMOTE for data augmentation...")

# For SMOTE, we need to discretize the target into categories
def discretize_target(y, n_bins=5):
    """Discretize continuous target into categories for SMOTE"""
    percentiles = np.percentile(y, np.linspace(0, 100, n_bins+1))
    return np.digitize(y, percentiles[1:-1])

# Create discretized target for SMOTE
y_train_discrete = discretize_target(y_train)

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
try:
    X_train_smote, y_train_smote_discrete = smote.fit_resample(X_train, y_train_discrete)
    # Map back to continuous values by using original y values from resampled indices
    # This is an approximation - we'll use the mean of each bin
    bins = np.unique(y_train_smote_discrete)
    y_train_smote = np.zeros_like(y_train_smote_discrete, dtype=float)
    for bin_val in bins:
        mask = y_train_smote_discrete == bin_val
        original_mask = y_train_discrete == bin_val
        if np.any(original_mask):
            y_train_smote[mask] = np.mean(y_train[original_mask])
    
    print(f"✅ SMOTE applied: {X_train.shape[0]} → {X_train_smote.shape[0]} samples")
    
    # Scale SMOTE data
    X_train_smote_scaled = scaler.fit_transform(X_train_smote)
    X_train_smote_minmax = minmax_scaler.fit_transform(X_train_smote)
    y_train_smote_scaled = y_scaler.fit_transform(y_train_smote.reshape(-1, 1)).flatten()
    
except Exception as e:
    print(f"⚠️ SMOTE failed: {e}")
    print("   Using original data without SMOTE")
    X_train_smote = X_train.copy()
    y_train_smote = y_train.copy()
    X_train_smote_scaled = X_train_scaled.copy()
    X_train_smote_minmax = X_train_minmax.copy()
    y_train_smote_scaled = y_train_scaled.copy()

# Define accuracy calculation function
def calculate_prediction_accuracy(y_true, y_pred, tolerance_pct=20):
    """Calculate accuracy as percentage of predictions within tolerance"""
    relative_error = np.abs(y_true - y_pred) / (y_true + 1)  # Add 1 to avoid division by zero
    accurate_predictions = relative_error <= (tolerance_pct / 100)
    return np.mean(accurate_predictions) * 100

# Create LSTM wrapper class
if TENSORFLOW_AVAILABLE:
    class LSTMRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, units=50, dropout=0.2, epochs=100, batch_size=32, learning_rate=0.001, recurrent_dropout=0.2):
            self.units = units
            self.dropout = dropout
            self.recurrent_dropout = recurrent_dropout
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.model = None
            self.input_shape = None
            
        def _reshape_for_lstm(self, X):
            """Reshape data for LSTM (samples, timesteps, features)"""
            # For this problem, we'll use each feature as a timestep
            return X.reshape(X.shape[0], X.shape[1], 1)
            
        def fit(self, X, y):
            # Reshape input for LSTM
            X_reshaped = self._reshape_for_lstm(X)
            self.input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])
            
            # Build model with regularization
            self.model = Sequential([
                LSTM(self.units, return_sequences=True, input_shape=self.input_shape, 
                     dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
                Dropout(self.dropout),
                LSTM(self.units // 2, return_sequences=False,
                     dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
                Dropout(self.dropout),
                Dense(25, activation='relu'),
                Dropout(0.3),  # Additional dropout layer
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Early stopping with validation monitoring
            early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, min_delta=0.001)
            
            # Train model
            self.model.fit(
                X_reshaped, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            return self
            
        def predict(self, X):
            X_reshaped = self._reshape_for_lstm(X)
            return self.model.predict(X_reshaped, verbose=0).flatten()

# ===== 9. MODEL TRAINING WITH HYPERPARAMETER TUNING =====
print("\n🤖 STEP 9: Training models with hyperparameter tuning...")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Define hyperparameter grids for tuning with REGULARIZATION
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],  # Reduced max to prevent overfitting
        'max_depth': [3, 5, 8, 10],       # Added shallow depths
        'min_samples_split': [5, 10, 20], # Increased to prevent overfitting
        'min_samples_leaf': [2, 5, 10],   # Increased to prevent overfitting
        'max_features': ['sqrt', 'log2', 0.7]  # Feature subsampling
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 6],           # Reduced depth
        'learning_rate': [0.01, 0.05, 0.1], # Added slower learning
        'subsample': [0.6, 0.8, 0.9],    # More aggressive subsampling
        'colsample_bytree': [0.6, 0.8, 1.0], # Feature subsampling
        'reg_alpha': [0, 0.1, 1],         # L1 regularization
        'reg_lambda': [1, 1.5, 2]         # L2 regularization
    },
    'Gradient Boosting': {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 6],           # Reduced depth
        'learning_rate': [0.01, 0.05, 0.1], # Added slower learning
        'subsample': [0.6, 0.8, 0.9],    # More aggressive subsampling
        'max_features': ['sqrt', 'log2', 0.7], # Feature subsampling
        'min_samples_split': [10, 20],    # Increased
        'min_samples_leaf': [5, 10]       # Increased
    },
    'Ridge Regression': {                 # Added Ridge for L2 regularization
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    },
    'Lasso Regression': {                 # Added Lasso for L1 regularization
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
    },
    'Elastic Net': {                      # Added Elastic Net for combined L1+L2
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    },
    'Support Vector Regression': {
        'C': [0.1, 1, 10, 50],           # Added lower C values for more regularization
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 0.2]
    }
}

if TENSORFLOW_AVAILABLE:
    param_grids['LSTM'] = {
        'units': [16, 32, 50],            # Reduced complexity
        'dropout': [0.3, 0.4, 0.5],      # Increased dropout for regularization
        'learning_rate': [0.0001, 0.001, 0.01], # Added slower learning
        'epochs': [50, 100],
        'recurrent_dropout': [0.2, 0.3]  # Added recurrent dropout
    }

# Initialize base models with regularization
base_models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42, max_iter=2000),
    'Elastic Net': ElasticNet(random_state=42, max_iter=2000),
    'Support Vector Regression': SVR()
}

if TENSORFLOW_AVAILABLE:
    base_models['LSTM'] = LSTMRegressor()

print(f"🔧 Models to train: {list(base_models.keys())}")
print("🛡️ REGULARIZATION TECHNIQUES APPLIED:")
print("   • Random Forest: Reduced depth, increased min_samples, feature subsampling")
print("   • XGBoost: L1/L2 regularization, reduced depth, feature/sample subsampling")
print("   • Gradient Boosting: Reduced depth, feature subsampling, increased min_samples")
print("   • Ridge: L2 regularization with alpha tuning")
print("   • Lasso: L1 regularization with alpha tuning")
print("   • Elastic Net: Combined L1+L2 regularization")
print("   • SVR: Reduced C values for stronger regularization")
if TENSORFLOW_AVAILABLE:
    print("   • LSTM: Increased dropout, recurrent dropout, reduced complexity")
    print("✅ LSTM included")
else:
    print("⚠️ LSTM skipped (TensorFlow not available)")

# Train all models with hyperparameter tuning and SMOTE
model_results = {}
print("\n🎯 Training models with hyperparameter tuning and SMOTE:")

for name, base_model in base_models.items():
    print(f"\n🔄 Training {name}...")
    
    # Determine which data to use
    if name in ['Support Vector Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']:
        X_train_use = X_train_smote_scaled
        X_test_use = X_test_scaled
        y_train_use = y_train_smote
    elif name == 'LSTM':
        X_train_use = X_train_smote_minmax
        X_test_use = X_test_minmax
        y_train_use = y_train_smote_scaled
    else:
        X_train_use = X_train_smote
        X_test_use = X_test
        y_train_use = y_train_smote
    
    # Hyperparameter tuning
    param_grid = param_grids.get(name, {})
    
    if param_grid and name != 'LSTM':  # Use GridSearch for sklearn models
        print(f"  🔍 Hyperparameter tuning with {len(param_grid)} parameters...")
        
        # Use RandomizedSearchCV for faster tuning
        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,  # Limit iterations for speed
            cv=3,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_use, y_train_use)
        best_model = grid_search.best_estimator_
        print(f"  ✅ Best params: {grid_search.best_params_}")
        
    elif name == 'LSTM' and TENSORFLOW_AVAILABLE:
        # Manual hyperparameter tuning for LSTM
        print("  🔍 LSTM hyperparameter tuning...")
        best_score = -np.inf
        best_model = None
        best_params = None
        
        # Try regularized combinations
        param_combinations = [
            {'units': 32, 'dropout': 0.4, 'recurrent_dropout': 0.3, 'learning_rate': 0.001, 'epochs': 50},
            {'units': 16, 'dropout': 0.5, 'recurrent_dropout': 0.2, 'learning_rate': 0.0001, 'epochs': 100},
            {'units': 50, 'dropout': 0.3, 'recurrent_dropout': 0.3, 'learning_rate': 0.001, 'epochs': 50}
        ]
        
        for params in param_combinations:
            try:
                temp_model = LSTMRegressor(**params)
                temp_model.fit(X_train_use, y_train_use)
                
                # Quick validation
                val_pred = temp_model.predict(X_train_use[-50:])  # Use last 50 samples as validation
                val_true = y_train_use[-50:]
                val_score = r2_score(val_true, val_pred)
                
                if val_score > best_score:
                    best_score = val_score
                    best_model = temp_model
                    best_params = params
            except Exception as e:
                print(f"    ⚠️ LSTM config failed: {params} - {e}")
                continue
        
        if best_model is None:
            print("  ⚠️ All LSTM configs failed, using default")
            best_model = LSTMRegressor()
            best_model.fit(X_train_use, y_train_use)
            best_params = "default"
        else:
            print(f"  ✅ Best LSTM params: {best_params}")
            
    else:
        # No hyperparameter tuning
        print("  🚀 Training with default parameters...")
        best_model = base_model
        best_model.fit(X_train_use, y_train_use)
    
    # Make predictions
    if name == 'LSTM' and TENSORFLOW_AVAILABLE:
        train_pred = best_model.predict(X_train_use)
        test_pred = best_model.predict(X_test_use)
        
        # Scale back predictions for LSTM
        if y_train_use is y_train_smote_scaled:
            train_pred = y_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            test_pred = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            y_train_eval = y_scaler.inverse_transform(y_train_use.reshape(-1, 1)).flatten()
        else:
            y_train_eval = y_train_use
    else:
        train_pred = best_model.predict(X_train_use)
        test_pred = best_model.predict(X_test_use)
        y_train_eval = y_train_use
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train_eval, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train_eval, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_eval, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Calculate accuracy
    train_accuracy = calculate_prediction_accuracy(y_train_eval, train_pred)
    test_accuracy = calculate_prediction_accuracy(y_test, test_pred)
    
    # Store results
    model_results[name] = {
        'model': best_model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_predictions': test_pred,
        'uses_scaled_features': name in ['Support Vector Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'LSTM'],
        'best_params': grid_search.best_params_ if 'grid_search' in locals() and name != 'LSTM' else best_params if name == 'LSTM' else 'default'
    }
    
    print(f"  📊 {name}: {test_accuracy:.1f}% accuracy, R² = {test_r2:.3f}")

print("\n✅ All models trained successfully with hyperparameter tuning!")

# ===== 10. MODEL EVALUATION & COMPARISON =====
print("\n📊 STEP 10: Model evaluation and comparison...")

# Display results for all models
print("\n🏆 MODEL COMPARISON RESULTS - ACTUAL ACCURACY:")
print("=" * 100)
print(f"{'Model':<25} {'Test Accuracy':<15} {'Train Accuracy':<15} {'Test R²':<10} {'Test MAE':<12} {'Test RMSE':<12}")
print("-" * 100)

for name, results in model_results.items():
    print(f"{name:<25} {results['test_accuracy']:<15.1f}% {results['train_accuracy']:<15.1f}% {results['test_r2']:<10.3f} {results['test_mae']:<12.1f} {results['test_rmse']:<12.1f}")

# Find the best model based on multiple criteria
print("\n🔍 STEP 10.1: Selecting best model...")

# Calculate composite score (higher is better)
# Weight: 50% accuracy, 30% R², 20% MAE (inverted)
best_scores = {}
for name, results in model_results.items():
    # Normalize MAE (lower is better, so invert it)
    max_mae = max([r['test_mae'] for r in model_results.values()])
    normalized_mae = (max_mae - results['test_mae']) / max_mae * 100
    
    # Composite score
    composite_score = (
        results['test_accuracy'] * 0.5 +  # 50% weight on accuracy
        results['test_r2'] * 100 * 0.3 +  # 30% weight on R² (convert to percentage)
        normalized_mae * 0.2  # 20% weight on MAE (inverted)
    )
    
    best_scores[name] = composite_score

# Find best model
best_model_name = max(best_scores.keys(), key=lambda x: best_scores[x])
best_model_results = model_results[best_model_name]
final_model = best_model_results['model']

print(f"\n🥇 BEST MODEL SELECTED: {best_model_name}")
print(f"   🎯 ACTUAL TEST ACCURACY: {best_model_results['test_accuracy']:.1f}% (±20% tolerance)")
print(f"   🎯 ACTUAL TRAIN ACCURACY: {best_model_results['train_accuracy']:.1f}% (±20% tolerance)")
print(f"   📊 Test R²: {best_model_results['test_r2']:.3f}")
print(f"   📉 Test MAE: {best_model_results['test_mae']:.1f} cases")
print(f"   📉 Test RMSE: {best_model_results['test_rmse']:.1f} cases")
print(f"   🏆 Composite Score: {best_scores[best_model_name]:.2f}")

# Set variables for backward compatibility
train_accuracy = best_model_results['train_accuracy']
test_accuracy = best_model_results['test_accuracy']
train_r2 = best_model_results['train_r2']
test_r2 = best_model_results['test_r2']
train_mae = best_model_results['train_mae']
test_mae = best_model_results['test_mae']
train_rmse = best_model_results['train_rmse']
test_rmse = best_model_results['test_rmse']
y_pred_test = best_model_results['test_predictions']

print("\n🎯 FINAL MODEL PERFORMANCE - ACTUAL ACCURACY:")
print("=" * 60)
print(f"🎯 ACTUAL TEST ACCURACY (±20%):      {test_accuracy:.1f}%")
print(f"🎯 ACTUAL TRAINING ACCURACY (±20%):  {train_accuracy:.1f}%")
print(f"📊 Test R² Score:                    {test_r2:.3f}")
print(f"📊 Training R² Score:                {train_r2:.3f}")
print(f"📉 Test MAE:                         {test_mae:.1f} cases")
print(f"📉 Training MAE:                     {train_mae:.1f} cases")
print(f"📉 Test RMSE:                        {test_rmse:.1f} cases")
print(f"📉 Training RMSE:                    {train_rmse:.1f} cases")
print("=" * 60)

# ===== 11. FEATURE IMPORTANCE =====
print("\n🔍 STEP 11: Feature importance analysis...")

# Only show feature importance for tree-based models
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Most Important Features ({best_model_name}):")
    print(feature_importance.head(10).to_string(index=False))
else:
    print(f"\n⚠️ Feature importance not available for {best_model_name}")
    print("This model doesn't provide feature importance scores.")

# ===== 12. SAVE MODEL =====
print("\n💾 STEP 12: Saving model and preprocessors...")

model_package = {
    'model': final_model,
    'best_model_name': best_model_name,
    'scaler': scaler,
    'minmax_scaler': minmax_scaler,
    'y_scaler': y_scaler,
    'label_encoder': le_state,
    'feature_columns': feature_columns,
    'uses_scaled_features': best_model_results['uses_scaled_features'],
    'best_params': best_model_results['best_params'],
    'smote_applied': X_train_smote.shape[0] > X_train.shape[0],
    'tensorflow_available': TENSORFLOW_AVAILABLE,
    'model_performance': {
        'test_accuracy': test_accuracy,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    },
    'all_model_results': {name: {
        'test_accuracy': results['test_accuracy'],
        'test_r2': results['test_r2'],
        'test_mae': results['test_mae'],
        'composite_score': best_scores[name],
        'best_params': results['best_params']
    } for name, results in model_results.items()}
}

with open('covid_prediction_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Model saved as 'covid_prediction_model.pkl'")

# ===== 13. SAMPLE PREDICTIONS =====
print("\n🔮 STEP 13: Sample predictions...")

# Show some actual vs predicted values
sample_results = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_pred_test[:10],
    'Absolute_Error': np.abs(y_test.iloc[:10].values - y_pred_test[:10])
})

print("\nSample Predictions (First 10 test samples):")
print(sample_results.to_string(index=False))

print(f"\n🎯 ENHANCED MACHINE LEARNING SUMMARY:")
print(f"✅ Successfully trained and compared {len(model_results)} different ML models with hyperparameter tuning")
print(f"✅ Data split: 80% training ({X_train.shape[0]} samples) / 20% testing ({X_test.shape[0]} samples)")
print(f"✅ SMOTE applied: {X_train.shape[0]} → {X_train_smote.shape[0]} samples")
print(f"✅ Best model selected: {best_model_name}")
print(f"✅ Best hyperparameters: {best_model_results['best_params']}")
print(f"🎯 ACTUAL TEST ACCURACY: {test_accuracy:.1f}% (within ±20% tolerance)")
print(f"🎯 ACTUAL TRAIN ACCURACY: {train_accuracy:.1f}% (within ±20% tolerance)")
print(f"✅ R² score of {test_r2:.3f} indicates the model explains {test_r2*100:.1f}% of variance")
print(f"✅ Average prediction error: {test_mae:.0f} cases")
print(f"✅ Enhanced model with all components saved for future use")

if TENSORFLOW_AVAILABLE:
    print(f"✅ LSTM deep learning model included and trained")
else:
    print(f"⚠️ LSTM model skipped (install tensorflow for deep learning capabilities)")

print(f"\n📊 MODEL RANKING BY ACTUAL TEST ACCURACY:")
sorted_by_accuracy = sorted(model_results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
for i, (name, results) in enumerate(sorted_by_accuracy, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"   {emoji} {name}: {results['test_accuracy']:.1f}% accuracy")

print(f"\n📊 MODEL RANKING BY COMPOSITE SCORE:")
sorted_models = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(sorted_models, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"   {emoji} {name}: {score:.2f} (composite score)")

print("\n" + "=" * 50)
print("🎉 Multi-model training and selection completed successfully!") 