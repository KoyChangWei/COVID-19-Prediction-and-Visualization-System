from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

# Try to import TensorFlow for LSTM model support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
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
            pass  # Not needed for prediction
            
        def predict(self, X):
            X_reshaped = self._reshape_for_lstm(X)
            return self.model.predict(X_reshaped, verbose=0).flatten()
            
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)

# Load the trained model and results
try:
    with open('covid_prediction_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model components with correct keys
    best_model = model_data['model']
    all_results = model_data.get('all_model_results', {})
    best_model_name = model_data['best_model_name']
    scaler = model_data.get('scaler')
    minmax_scaler = model_data.get('minmax_scaler')
    y_scaler = model_data.get('y_scaler')
    label_encoder = model_data.get('label_encoder')
    feature_columns = model_data.get('feature_columns', [])
    uses_scaled_features = model_data.get('uses_scaled_features', False)
    
    print(f"✅ Loaded model: {best_model_name}")
    print(f"✅ Available models: {list(all_results.keys())}")
    print(f"✅ Uses scaled features: {uses_scaled_features}")
    print(f"✅ Feature columns: {len(feature_columns)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    best_model = None
    all_results = {}
    best_model_name = "Model Loading Error"
    scaler = None
    minmax_scaler = None
    y_scaler = None
    label_encoder = None
    feature_columns = []
    uses_scaled_features = False

# Malaysian states list (matching the label encoder)
MALAYSIAN_STATES = [
    'Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 
    'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 
    'Sarawak', 'Selangor', 'Terengganu', 'W.P. Kuala Lumpur', 
    'W.P. Labuan', 'W.P. Putrajaya'
]

def calculate_trend_analysis(current_total, prediction):
    """Calculate trend analysis and indicators"""
    try:
        if current_total == 0:
            change_percentage = 0
        else:
            change_percentage = ((prediction - current_total) / current_total) * 100
        
        # Determine trend
        if change_percentage > 10:
            trend = "📈"  # Strong increase
        elif change_percentage > 5:
            trend = "📊"  # Moderate increase
        elif change_percentage > -5:
            trend = "➡️"  # Stable
        elif change_percentage > -10:
            trend = "📉"  # Moderate decrease
        else:
            trend = "⬇️"  # Strong decrease
            
        return {
            'trend': trend,
            'change_percentage': round(change_percentage, 1)
        }
    except:
        return {
            'trend': "❓",
            'change_percentage': 0
        }

def validate_input_data(data):
    """Validate and sanitize input data"""
    try:
        # Required fields
        required_fields = ['state', 'month', 'year', 'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost']
        
        for field in required_fields:
            if field not in data or data[field] == '':
                return False, f"Missing required field: {field}"
        
        # Validate state
        if data['state'] not in MALAYSIAN_STATES:
            return False, "Invalid state selected"
        
        # Validate numeric fields
        try:
            month = int(data['month'])
            year = int(data['year'])
            cases_unvax = int(data['cases_unvax'])
            cases_pvax = int(data['cases_pvax'])
            cases_fvax = int(data['cases_fvax'])
            cases_boost = int(data['cases_boost'])
        except ValueError:
            return False, "Invalid numeric values"
        
        # Validate ranges
        if not (1 <= month <= 12):
            return False, "Month must be between 1 and 12"
        
        if not (2020 <= year <= 2030):
            return False, "Year must be between 2020 and 2030"
        
        if any(cases < 0 for cases in [cases_unvax, cases_pvax, cases_fvax, cases_boost]):
            return False, "Case numbers cannot be negative"
        
        total_cases = cases_unvax + cases_pvax + cases_fvax + cases_boost
        if total_cases == 0:
            return False, "Total cases cannot be zero"
        
        if total_cases > 100000:
            return False, "Total cases seem unrealistically high (>100,000)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@app.route('/')
def index():
    """Main page with all sections"""
    # Use states from label encoder if available, otherwise fallback to our list
    states_list = list(label_encoder.classes_) if label_encoder else MALAYSIAN_STATES
    return render_template('index.html', 
                         states=states_list,
                         all_results=all_results,
                         model_name=best_model_name)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with comprehensive error handling"""
    try:
        # Check if model is loaded
        if best_model is None:
            return jsonify({
                'success': False,
                'error': 'Prediction model is not available. Please check server logs.'
            })
        
        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            })
        
        # Validate input
        is_valid, validation_message = validate_input_data(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Invalid input: {validation_message}'
            })
        
        # Extract and convert data
        state = data['state']
        month = int(data['month'])
        year = int(data['year'])
        cases_unvax = int(data['cases_unvax'])
        cases_pvax = int(data['cases_pvax'])
        cases_fvax = int(data['cases_fvax'])
        cases_boost = int(data['cases_boost'])
        
        # Calculate additional metrics
        total_cases = cases_unvax + cases_pvax + cases_fvax + cases_boost
        vaccinated_cases = cases_pvax + cases_fvax + cases_boost
        vaccination_rate = round((vaccinated_cases / total_cases) * 100, 1) if total_cases > 0 else 0
        
        # Prepare features for prediction using the original model structure
        try:
            # Encode state using the label encoder
            if label_encoder:
                state_encoded = label_encoder.transform([state])[0]
            else:
                # Fallback: use state index
                state_encoded = MALAYSIAN_STATES.index(state) if state in MALAYSIAN_STATES else 0
            
            # Calculate quarter and holiday season
            quarter = (month - 1) // 3 + 1
            is_holiday = 1 if month in [12, 1, 6, 7] else 0
            
            # Create feature vector matching the training format
            features = np.array([[
                cases_unvax,
                cases_pvax, 
                cases_fvax,
                cases_boost,
                total_cases,
                total_cases,  # lag_1_total (using current as approximation)
                total_cases,  # lag_2_total
                total_cases,  # lag_3_total
                cases_unvax,  # lag_1_unvax
                cases_pvax,   # lag_1_pvax
                cases_fvax,   # lag_1_fvax
                cases_boost,  # lag_1_boost
                total_cases,  # rolling_3m_avg
                total_cases,  # rolling_6m_avg
                0,            # cases_change_1m
                0,            # cases_change_3m
                vaccination_rate / 100,  # vax_rate (as decimal)
                month,
                quarter,
                year,
                is_holiday,
                state_encoded
            ]])
            
            # Scale features if needed
            if uses_scaled_features and scaler is not None:
                if best_model_name == 'LSTM' and minmax_scaler is not None:
                    features = minmax_scaler.transform(features)
                else:
                    features = scaler.transform(features)
            
            # Make prediction
            prediction = best_model.predict(features)[0]
            
            # For LSTM, scale back the prediction if needed
            if best_model_name == 'LSTM' and y_scaler is not None:
                try:
                    prediction = y_scaler.inverse_transform([[prediction]])[0][0]
                except:
                    pass
            
            # Ensure prediction is reasonable
            if prediction < 0:
                prediction = 0
            elif prediction > 1000000:  # Cap at 1M cases
                prediction = 1000000
            
            prediction = int(round(prediction))
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            print(f"❌ Feature shape: {features.shape if 'features' in locals() else 'Not created'}")
            print(f"❌ Model type: {type(best_model)}")
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            })
        
        # Calculate trend analysis
        trend_data = calculate_trend_analysis(total_cases, prediction)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'current_total': total_cases,
            'vax_rate': vaccination_rate,
            'trend': trend_data['trend'],
            'change_percentage': trend_data['change_percentage'],
            'model_used': best_model_name,
            'state': state,
            'input_data': {
                'month': month,
                'year': year,
                'cases_breakdown': {
                    'unvaccinated': cases_unvax,
                    'partially_vaccinated': cases_pvax,
                    'fully_vaccinated': cases_fvax,
                    'boosted': cases_boost
                }
            },
            'analysis': {
                'total_current_cases': total_cases,
                'vaccination_coverage': f"{vaccination_rate}%",
                'predicted_change': f"{trend_data['change_percentage']:+.1f}%"
            }
        }
        
        print(f"✅ Prediction successful: {total_cases} → {prediction} cases ({trend_data['change_percentage']:+.1f}%)")
        return jsonify(response)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Prediction error: {error_details}")
        
        return jsonify({
            'success': False,
            'error': f'Server error during prediction. Please try again. ({str(e)})'
        })

@app.route('/api/states')
def get_states():
    """API endpoint to get available states"""
    return jsonify({
        'success': True,
        'states': MALAYSIAN_STATES
    })

@app.route('/api/model-info')
def get_model_info():
    """API endpoint to get model information"""
    return jsonify({
        'success': True,
        'best_model': best_model_name,
        'all_models': list(all_results.keys()),
        'model_performance': {
            model_name: {
                'accuracy': round(results.get('test_accuracy', 0), 2),
                'r2_score': round(results.get('test_r2', 0), 3),
                'mae': round(results.get('test_mae', 0), 2)
            }
            for model_name, results in all_results.items()
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': best_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # For local development and production
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)

# For production deployment (Render will use this)
app.config['ENV'] = 'production'