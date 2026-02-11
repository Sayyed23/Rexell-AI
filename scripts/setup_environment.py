# setup_environment.py
import subprocess
import sys

def install_requirements():
    """Install all required packages"""
    packages = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',
        'catboost>=1.0.0',
        'torch>=1.10.0',
        'pytorch-tabnet>=3.1.0',
        'tensorflow>=2.8.0',
        'imbalanced-learn>=0.9.0',
        'optuna>=2.10.0',
        'hyperopt>=0.2.7',
        'scikit-optimize>=0.9.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'plotly>=5.6.0',
        'jupyter',
        'joblib>=1.1.0',
        'mlflow>=1.24.0',
        'shap>=0.40.0'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except:
            print(f"⚠️  Failed to install: {package}")
    
    print("\n🎉 Environment setup complete!")

if __name__ == "__main__":
    install_requirements()