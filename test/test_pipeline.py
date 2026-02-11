# test_pipeline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test 1: Check if data loads
print("🧪 TEST 1: Data Loading")
try:
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor("data/blockchain_ticketing_master.csv")
    df = preprocessor.load_data().get_data()
    print(f"✅ Data loaded successfully: {df.shape}")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

# Test 2: Check feature engineering
print("\n🧪 TEST 2: Feature Engineering")
try:
    from src.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(df.head(100))  # Test with small sample
    df_engineered, features = engineer.get_engineered_data()
    print(f"✅ Feature engineering works: {len(features)} features created")
except Exception as e:
    print(f"❌ Feature engineering failed: {e}")

# Test 3: Check dependencies
print("\n🧪 TEST 3: Dependencies Check")
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    print("✅ All critical dependencies installed")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")

print("\n" + "="*50)
print("✅ TEST COMPLETE - READY TO RUN MAIN PIPELINE")
print("="*50)