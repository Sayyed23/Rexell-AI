# src/data_preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load and basic info"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"📊 Dataset loaded: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Check if scalping_label exists
            if 'scalping_label' in self.df.columns:
                print(f"🎯 Target distribution:\n{self.df['scalping_label'].value_counts(normalize=True)}")
            
            return self
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise e
    
    def handle_missing_values(self):
        """Handle missing values intelligently"""
        print("\n🔍 Handling missing values...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Fill missing values based on column type
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                # For numerical columns
                if self.df[column].dtype in ['int64', 'float64']:
                    if column in ['price_paid', 'original_event_price', 'risk_score']:
                        self.df[column] = self.df[column].fillna(self.df[column].median())
                    else:
                        self.df[column] = self.df[column].fillna(0)
                # For categorical columns
                elif self.df[column].dtype == 'object':
                    self.df[column] = self.df[column].fillna('unknown')
        
        print(f"✅ Missing values handled")
        return self
    
    def detect_outliers(self):
        """Detect and handle outliers in numerical columns"""
        print("\n🔍 Detecting outliers...")
        
        # Identify numerical columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        for col in numerical_cols:
            if col in ['scalping_label', 'fraud_label']:  # Skip target columns
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        print("✅ Outliers handled (capped)")
        return self
    
    def get_data(self):
        """Return processed dataframe"""
        return self.df