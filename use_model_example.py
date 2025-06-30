import pandas as pd
import numpy as np
import pickle

# Try to import TensorFlow and define LSTM class for model loading
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.base import BaseEstimator, RegressorMixin
    
    class LSTMRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, units=50, dropout=0.2, epochs=100, batch_size=32, learning_rate=0.001):
            self.units = units
            self.dropout = dropout
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.model = None
            self.input_shape = None
            
        def _reshape_for_lstm(self, X):
            return X.reshape(X.shape[0], X.shape[1], 1)
            
        def fit(self, X, y):
            # Implementation not needed for loading, just placeholder
            pass
            
        def predict(self, X):
            X_reshaped = self._reshape_for_lstm(X)
            return self.model.predict(X_reshaped, verbose=0).flatten()
            
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - LSTM functionality limited")

print("🔮 COVID-19 Case Prediction - Using Trained Model")
print("=" * 50)

# Load the saved model
print("📥 Loading trained model...")
with open('covid_prediction_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
best_model_name = model_package['best_model_name']
scaler = model_package['scaler']
minmax_scaler = model_package['minmax_scaler']
y_scaler = model_package['y_scaler']
label_encoder = model_package['label_encoder']
feature_columns = model_package['feature_columns']
uses_scaled_features = model_package['uses_scaled_features']
best_params = model_package['best_params']
smote_applied = model_package['smote_applied']
tensorflow_available = model_package['tensorflow_available']
performance = model_package['model_performance']
all_results = model_package['all_model_results']

print("✅ Enhanced model loaded successfully!")
print(f"🏆 Best Model: {best_model_name}")
print(f"🎯 Best Parameters: {best_params}")
print(f"📊 Model Performance: {performance['test_accuracy']:.1f}% accuracy, R² = {performance['test_r2']:.3f}")
print(f"🔄 SMOTE Applied: {'Yes' if smote_applied else 'No'}")
print(f"🧠 Deep Learning: {'Available' if tensorflow_available else 'Not Available'}")

# Show all model comparison
print(f"\n📊 Complete Model Comparison Results:")
print("-" * 70)
for name, results in all_results.items():
    emoji = "🏆" if name == best_model_name else "  "
    print(f"{emoji} {name:<25}: {results['test_accuracy']:>5.1f}% acc, R²={results['test_r2']:>5.3f}, Score={results['composite_score']:>5.1f}")
    if results['best_params'] != 'default':
        print(f"    Best params: {results['best_params']}")
print("-" * 70)

# Load and prepare some sample data for prediction
print("\n📊 Preparing sample data for prediction...")
df = pd.read_csv('covid_cases_vaxstatus.csv')
df['date'] = pd.to_datetime(df['date'])

# Take the latest available data for each state as an example
df_latest = df[df['state'].str.lower() != 'malaysia'].copy()
df_latest['month_year'] = df_latest['date'].dt.to_period('M').astype(str)

# Get the most recent month's data for each state
latest_data = df_latest.groupby('state').apply(lambda x: x[x['month_year'] == x['month_year'].max()]).reset_index(drop=True)

print(f"Using latest data from: {latest_data['month_year'].iloc[0]}")
print(f"Available states: {list(latest_data['state'].unique())}")

# Function to make prediction for a specific state
def predict_next_month_cases(state_name, current_cases_data):
    """
    Predict next month's COVID cases for a specific state
    
    Parameters:
    state_name: Name of the state
    current_cases_data: Dictionary with current month's case data
    
    Returns:
    Predicted number of cases for next month
    """
    try:
        # Encode state
        state_encoded = label_encoder.transform([state_name])[0]
        
        # Create feature vector (this is a simplified example)
        # In practice, you'd need the full feature engineering pipeline
        features = np.array([[
            current_cases_data['cases_unvax'],
            current_cases_data['cases_pvax'], 
            current_cases_data['cases_fvax'],
            current_cases_data['cases_boost'],
            current_cases_data['total_cases'],
            current_cases_data.get('lag_1_total', current_cases_data['total_cases']),
            current_cases_data.get('lag_2_total', current_cases_data['total_cases']),
            current_cases_data.get('lag_3_total', current_cases_data['total_cases']),
            current_cases_data.get('lag_1_unvax', current_cases_data['cases_unvax']),
            current_cases_data.get('lag_1_pvax', current_cases_data['cases_pvax']),
            current_cases_data.get('lag_1_fvax', current_cases_data['cases_fvax']),
            current_cases_data.get('lag_1_boost', current_cases_data['cases_boost']),
            current_cases_data.get('rolling_3m_avg', current_cases_data['total_cases']),
            current_cases_data.get('rolling_6m_avg', current_cases_data['total_cases']),
            current_cases_data.get('cases_change_1m', 0),
            current_cases_data.get('cases_change_3m', 0),
            current_cases_data.get('vax_rate', 0.5),
            current_cases_data.get('month', 1),
            current_cases_data.get('quarter', 1),
            current_cases_data.get('year', 2023),
            current_cases_data.get('is_holiday_season', 0),
            state_encoded
        ]])
        
        # Scale features if needed
        if uses_scaled_features:
            if state_name == 'LSTM':  # LSTM uses MinMax scaling
                features = minmax_scaler.transform(features)
            else:
                features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # For LSTM, scale back the prediction if needed
        if best_model_name == 'LSTM' and hasattr(y_scaler, 'inverse_transform'):
            try:
                prediction = y_scaler.inverse_transform([[prediction]])[0][0]
            except:
                pass  # If scaling back fails, use original prediction
        
        return max(0, int(prediction))  # Ensure non-negative integer
        
    except Exception as e:
        print(f"Error making prediction for {state_name}: {e}")
        return None

# Example predictions for a few states
print("\n🔮 SAMPLE PREDICTIONS:")
print("=" * 30)

sample_states = ['Selangor', 'Johor', 'Kuala Lumpur'] if 'Selangor' in latest_data['state'].values else latest_data['state'].head(3).tolist()

for state in sample_states:
    if state in latest_data['state'].values:
        state_data = latest_data[latest_data['state'] == state].iloc[0]
        
        # Prepare current month's data
        current_data = {
            'cases_unvax': state_data['cases_unvax'],
            'cases_pvax': state_data['cases_pvax'],
            'cases_fvax': state_data['cases_fvax'], 
            'cases_boost': state_data['cases_boost'],
            'total_cases': state_data['cases_unvax'] + state_data['cases_pvax'] + state_data['cases_fvax'] + state_data['cases_boost'],
            'month': state_data['date'].month,
            'year': state_data['date'].year,
            'quarter': (state_data['date'].month - 1) // 3 + 1,
            'is_holiday_season': 1 if state_data['date'].month in [12, 1, 6, 7] else 0
        }
        
        prediction = predict_next_month_cases(state, current_data)
        
        print(f"\n🏢 {state}:")
        print(f"   Current month total cases: {current_data['total_cases']}")
        print(f"   Predicted next month: {prediction} cases")
        if prediction and current_data['total_cases'] > 0:
            change_pct = ((prediction - current_data['total_cases']) / current_data['total_cases']) * 100
            trend = "📈 Increase" if change_pct > 5 else "📉 Decrease" if change_pct < -5 else "➡️ Stable"
            print(f"   Expected change: {change_pct:+.1f}% {trend}")

print("\n" + "=" * 50)
print("💡 HOW TO USE THIS MODEL:")
print("1. Prepare current month's case data for any Malaysian state")
print("2. Use the predict_next_month_cases() function")
print("3. The model returns predicted cases for the next month")
print("\n⚠️ IMPORTANT NOTES:")
print(f"🏆 Best model selected: {best_model_name}")
print(f"📊 Model accuracy: ~{performance['test_accuracy']:.1f}% (within ±20% tolerance)")
print(f"🔧 {len(all_results)} different algorithms were compared with hyperparameter tuning")
print(f"🧠 Deep learning (LSTM) {'included' if tensorflow_available else 'not available'}")
print(f"🔄 SMOTE data augmentation {'applied' if smote_applied else 'not applied'}")
print("📈 Best used for trend analysis rather than exact predictions")
print("⚠️ COVID-19 cases are inherently unpredictable due to many external factors")
print("💡 Consider this as one input among many for decision-making")

print("\n✅ Model usage example completed!") 