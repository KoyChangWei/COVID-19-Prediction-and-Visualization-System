# Render Deployment Guide - COVID-19 Prediction System

## ✅ **FINAL WORKING SOLUTION**

The issue was that Render was using Python 3.13.4 (which doesn't have TensorFlow support) instead of the specified Python version. Here's the fix:

## 📁 **Files for Render Deployment**

### 1. `requirements.txt` (Working Version)
```
pandas>=1.5.3
numpy>=1.26.4
scikit-learn>=1.3.2
matplotlib>=3.7.3
seaborn>=0.13.2
xgboost>=1.7.5
imbalanced-learn>=0.11.0
tensorflow-cpu>=2.16.0
flask>=2.3.0
gunicorn>=21.0.0
```

### 2. `render.yaml` (Forces Python version via environment variable)
```yaml
services:
  - type: web
    name: covid-prediction-app
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --chdir prototype app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.11
```

### 3. `.python-version` (Backup version control)
```
3.12.11
```

### 4. `runtime.txt` (Secondary backup)
```
python-3.12.11
```

### 5. `Procfile` (Start command)
```
web: gunicorn --chdir prototype app:app
```

## 🚀 **Deployment Steps**

### Method 1: Using render.yaml (Recommended)
1. **Commit all files** to your repository
2. **In Render Dashboard**:
   - Create a **new service** from your GitHub repo
   - Render will automatically detect the `render.yaml` file
   - It will use the specified Python version and build commands

### Method 2: Manual Configuration
1. **In Render Dashboard**:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --chdir prototype app:app`
   - **Add environment variable**: `PYTHON_VERSION` = `3.12.11`

## 🔧 **Why This Works**

1. **Python 3.12.11**: TensorFlow supports Python 3.9-3.12, and 3.12.11 is stable
2. **Environment Variable Priority**: Render respects `PYTHON_VERSION` environment variable above all else
3. **tensorflow-cpu**: Uses CPU-only version which works perfectly on Render
4. **Compatible Dependencies**: All packages work with Python 3.12

## 📊 **Expected Behavior**

Your app will work with **FULL functionality**:
- ✅ XGBoost model (main predictor)
- ✅ Random Forest, Gradient Boosting, etc.
- ✅ **LSTM model (TensorFlow working!)**
- ✅ Web interface fully functional
- ✅ All 8 models available

## 🔍 **Troubleshooting**

If it still doesn't work:

1. **Delete the old service** and create a new one (Render sometimes caches settings)
2. **Check the build logs** for the Python version being used
3. **Verify environment variable** is set to `PYTHON_VERSION=3.12.11`

The combination of `render.yaml` + environment variable should force Render to use Python 3.12.11 instead of defaulting to 3.13.4. 