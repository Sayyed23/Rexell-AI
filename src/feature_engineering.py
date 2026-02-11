# src/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.features = []
        
    def create_time_features(self):
        """Create time-based features"""
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Time features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_of_month'] = self.df['timestamp'].dt.day
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour']/24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour']/24)
        
        self.features.extend(['hour', 'day_of_week', 'day_of_month', 'month', 
                             'is_weekend', 'hour_sin', 'hour_cos'])
        return self
    
    def create_price_features(self):
        """Create price-related features"""
        # Basic price features
        self.df['price_difference'] = self.df['price_paid'] - self.df['original_event_price']
        self.df['price_ratio'] = self.df['price_paid'] / (self.df['original_event_price'] + 1e-5)
        self.df['is_overpriced'] = (self.df['price_ratio'] > 2).astype(int)
        self.df['markup_category'] = pd.cut(self.df['markup_pct'], 
                                           bins=[-np.inf, 0, 50, 100, 200, np.inf], 
                                           labels=[0, 1, 2, 3, 4])
        
        self.features.extend(['price_difference', 'price_ratio', 'is_overpriced', 
                             'markup_category'])
        return self
    
    def create_user_behavior_features(self):
        """Create user-level behavioral features"""
        # Group by wallet address
        user_stats = self.df.groupby('wallet_address').agg({
            'transaction_hash': 'count',
            'ticket_count': ['sum', 'mean', 'std'],
            'price_paid': ['mean', 'std', 'max'],
            'event_id': 'nunique',
            'is_resale': 'sum',
            'timestamp': ['min', 'max']
        }).round(4)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        user_stats = user_stats.reset_index()
        
        # Rename columns
        user_stats.columns = [
            'wallet_address', 'user_total_transactions', 'user_total_tickets',
            'user_avg_tickets', 'user_std_tickets', 'user_avg_price',
            'user_std_price', 'user_max_price', 'user_unique_events',
            'user_resale_count', 'user_first_transaction', 'user_last_transaction'
        ]
        
        # Calculate additional features
        user_stats['user_resale_rate'] = user_stats['user_resale_count'] / user_stats['user_total_transactions']
        user_stats['user_transaction_frequency'] = (
            pd.to_datetime(user_stats['user_last_transaction']) - 
            pd.to_datetime(user_stats['user_first_transaction'])
        ).dt.days / user_stats['user_total_transactions']
        
        user_stats['user_transaction_frequency'] = user_stats['user_transaction_frequency'].replace([np.inf, -np.inf], 0)
        user_stats['user_transaction_frequency'] = user_stats['user_transaction_frequency'].fillna(0)
        
        # Merge with original dataframe
        self.df = self.df.merge(user_stats, on='wallet_address', how='left')
        
        self.features.extend([
            'user_total_transactions', 'user_total_tickets', 'user_avg_tickets',
            'user_std_tickets', 'user_avg_price', 'user_std_price', 'user_max_price',
            'user_unique_events', 'user_resale_count', 'user_resale_rate',
            'user_transaction_frequency'
        ])
        return self
    
    def create_transaction_features(self):
        """Create transaction-level features"""
        # Time since last transaction (per user)
        self.df = self.df.sort_values(['wallet_address', 'timestamp'])
        self.df['time_since_last_tx'] = self.df.groupby('wallet_address')['timestamp'].diff().dt.total_seconds()
        self.df['time_since_last_tx'] = self.df['time_since_last_tx'].fillna(0)
        
        # Rolling features (last 5 transactions per user)
        for window in [3, 5, 10]:
            self.df[f'avg_price_last_{window}'] = self.df.groupby('wallet_address')['price_paid']\
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            self.df[f'ticket_count_last_{window}'] = self.df.groupby('wallet_address')['ticket_count']\
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            
            self.features.extend([f'avg_price_last_{window}', f'ticket_count_last_{window}'])
        
        self.features.extend(['time_since_last_tx'])
        return self
    
    def create_aggregate_features(self):
        """Create aggregate features"""
        # Event-level features
        event_stats = self.df.groupby('event_id').agg({
            'price_paid': ['mean', 'std'],
            'ticket_count': 'sum',
            'wallet_address': 'nunique'
        }).round(4)
        
        event_stats.columns = ['event_avg_price', 'event_std_price', 
                              'event_total_tickets', 'event_unique_users']
        event_stats = event_stats.reset_index()
        
        self.df = self.df.merge(event_stats, on='event_id', how='left')
        
        # Price compared to event average
        self.df['price_vs_event_avg'] = self.df['price_paid'] / (self.df['event_avg_price'] + 1e-5)
        self.df['price_deviation'] = (self.df['price_paid'] - self.df['event_avg_price']) / (self.df['event_std_price'] + 1e-5)
        
        self.features.extend([
            'event_avg_price', 'event_std_price', 'event_total_tickets',
            'event_unique_users', 'price_vs_event_avg', 'price_deviation'
        ])
        return self
    
    def create_risk_features(self):
        """Create risk-based features"""
        # Composite risk score
        self.df['composite_risk'] = (
            self.df['risk_score'] * 0.3 +
            self.df['user_resale_rate'] * 0.2 +
            self.df['price_ratio'] * 0.2 +
            (self.df['user_total_transactions'] > 5).astype(int) * 0.15 +
            (self.df['ticket_count'] > 3).astype(int) * 0.15
        )
        
        # Risk categories
        self.df['risk_category'] = pd.cut(self.df['composite_risk'],
                                         bins=[0, 0.3, 0.6, 0.8, 1.0],
                                         labels=['low', 'medium', 'high', 'very_high'])
        
        self.features.extend(['composite_risk', 'risk_category'])
        return self
    
    def get_engineered_data(self):
        """Return engineered dataframe and feature list"""
        # Fill any remaining NaN
        self.df[self.features] = self.df[self.features].fillna(0)
        
        # One-hot encode categorical features
        categorical_features = [f for f in self.features if self.df[f].dtype == 'object']
        if categorical_features:
            self.df = pd.get_dummies(self.df, columns=categorical_features, drop_first=True)
            
            # Update features list
            new_categorical_cols = [col for col in self.df.columns 
                                  if any(f in col for f in categorical_features)]
            self.features = [f for f in self.features if f not in categorical_features]
            self.features.extend(new_categorical_cols)
        
        print(f"✅ Total features created: {len(self.features)}")
        return self.df, self.features

# Usage
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess
    preprocessor = DataPreprocessor('data/raw/transactions.csv')
    df = preprocessor.load_data().handle_missing_values().detect_outliers().df
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df_engineered, features = (engineer
                              .create_time_features()
                              .create_price_features()
                              .create_user_behavior_features()
                              .create_transaction_features()
                              .create_aggregate_features()
                              .create_risk_features()
                              .get_engineered_data())
    
    df_engineered.to_csv('data/processed/engineered_data.csv', index=False)
    print(f"💾 Engineered data saved with {len(features)} features")