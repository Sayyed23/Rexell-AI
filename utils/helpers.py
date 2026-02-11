# utils/helpers.py
import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def load_config(config_path="config/config.yaml"):
    """Load configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, path, name):
    """Save trained model"""
    create_directory(os.path.dirname(path))
    joblib.dump(model, os.path.join(path, f"{name}.pkl"))
    print(f"✅ Model saved: {name}.pkl")
    
def load_model(path):
    """Load trained model"""
    return joblib.load(path)

def save_results(results_df, path="results/model_comparison.csv"):
    """Save results dataframe"""
    create_directory(os.path.dirname(path))
    results_df.to_csv(path, index=False)
    print(f"✅ Results saved: {path}")
    
def print_bold(text):
    """Print text in bold"""
    print("\033[1m" + text + "\033[0m")
    
def timer(start_time, task_name=""):
    """Calculate and print elapsed time"""
    elapsed = datetime.now() - start_time
    print(f"⏱️  {task_name} completed in {elapsed.total_seconds():.2f} seconds")
    return elapsed