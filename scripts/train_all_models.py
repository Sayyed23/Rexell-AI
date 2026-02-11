# scripts/train_all_models.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all model classes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

# Ensemble
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
from sklearn.calibration import CalibratedClassifierCV

import joblib
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_score = 0
        
    def load_and_split_data(self, test_size=0.2, random_state=42):
        """Load data and create train-test split"""
        print("📊 Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop(['scalping_label', 'transaction_hash', 'wallet_address', 
                    'timestamp', 'fraud_label'], axis=1, errors='ignore')
        y = df['scalping_label']
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data loaded: {X.shape}")
        print(f"   Training: {self.X_train.shape}, Testing: {self.X_test.shape}")
        print(f"   Target distribution - Train: {self.y_train.value_counts().to_dict()}")
        
        return self
    
    def train_traditional_models(self):
        """Train all traditional ML models"""
        print("\n" + "="*60)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*60)
        
        traditional_models = {
            'Logistic_Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=8, random_state=42
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, random_state=42
            ),
            'SVM': SVC(
                probability=True, random_state=42, class_weight='balanced'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive_Bayes': GaussianNB()
        }
        
        for name, model in tqdm(traditional_models.items(), desc="Training traditional models"):
            try:
                start_time = time.time()
                model.fit(self.X_train_scaled, self.y_train)
                training_time = time.time() - start_time
                
                # Predictions
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Store model
                self.models[name] = model
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                metrics.update({
                    'Model': name,
                    'Training_Time': training_time,
                    'Category': 'Traditional_ML'
                })
                
                self.results.append(metrics)
                print(f"✅ {name}: F1={metrics['F1_Score']:.4f}, AUC={metrics['AUC_ROC']:.4f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
        
        return self
    
    def train_gradient_boosting_models(self):
        """Train gradient boosting models"""
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING MODELS")
        print("="*60)
        
        boosting_models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=len(self.y_train[self.y_train==0])/len(self.y_train[self.y_train==1])
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
                auto_class_weights='Balanced'
            )
        }
        
        for name, model in tqdm(boosting_models.items(), desc="Training boosting models"):
            try:
                start_time = time.time()
                
                if name == 'CatBoost':
                    # Handle categorical features for CatBoost
                    cat_features = np.where(self.X_train.nunique() < 10)[0]
                    model.fit(self.X_train, self.y_train, 
                            cat_features=cat_features if len(cat_features) > 0 else None)
                else:
                    model.fit(self.X_train_scaled, self.y_train)
                
                training_time = time.time() - start_time
                
                # Predictions
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Store model
                self.models[name] = model
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                metrics.update({
                    'Model': name,
                    'Training_Time': training_time,
                    'Category': 'Gradient_Boosting'
                })
                
                self.results.append(metrics)
                print(f"✅ {name}: F1={metrics['F1_Score']:.4f}, AUC={metrics['AUC_ROC']:.4f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
        
        return self
    
    def train_deep_learning_models(self):
        """Train deep learning models"""
        print("\n" + "="*60)
        print("TRAINING DEEP LEARNING MODELS")
        print("="*60)
        
        # 1. Multi-Layer Perceptron
        print("Training Neural Network (MLP)...")
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=300,
                random_state=42,
                early_stopping=True
            )
            
            start_time = time.time()
            mlp.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            
            y_pred = mlp.predict(self.X_test_scaled)
            y_pred_proba = mlp.predict_proba(self.X_test_scaled)[:, 1]
            
            self.models['Neural_Network_MLP'] = mlp
            
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics.update({
                'Model': 'Neural_Network_MLP',
                'Training_Time': training_time,
                'Category': 'Deep_Learning'
            })
            
            self.results.append(metrics)
            print(f"✅ Neural Network (MLP): F1={metrics['F1_Score']:.4f}")
            
        except Exception as e:
            print(f"❌ Neural Network failed: {str(e)}")
        
        # 2. TabNet
        print("Training TabNet...")
        try:
            tabnet = TabNetClassifier(
                n_d=8, n_a=8, n_steps=3,
                gamma=1.3, lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='entmax',
                scheduler_params={"step_size":10, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0,
                seed=42
            )
            
            start_time = time.time()
            tabnet.fit(
                self.X_train_scaled, self.y_train.values,
                eval_set=[(self.X_test_scaled, self.y_test.values)],
                max_epochs=50,
                patience=10,
                batch_size=1024,
                virtual_batch_size=128
            )
            training_time = time.time() - start_time
            
            y_pred = tabnet.predict(self.X_test_scaled)
            y_pred_proba = tabnet.predict_proba(self.X_test_scaled)[:, 1]
            
            self.models['TabNet'] = tabnet
            
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics.update({
                'Model': 'TabNet',
                'Training_Time': training_time,
                'Category': 'Deep_Learning'
            })
            
            self.results.append(metrics)
            print(f"✅ TabNet: F1={metrics['F1_Score']:.4f}")
            
        except Exception as e:
            print(f"❌ TabNet failed: {str(e)}")
        
        return self
    
    def train_ensemble_models(self):
        """Train ensemble models"""
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODELS")
        print("="*60)
        
        # Select top models for ensemble
        if len(self.models) >= 3:
            # Get top 3 models by F1 score
            results_df = pd.DataFrame(self.results)
            if not results_df.empty:
                top_models_info = results_df.nlargest(3, 'F1_Score')[['Model', 'F1_Score']]
                print(f"Top models for ensemble: {top_models_info['Model'].tolist()}")
                
                # Create estimators list
                estimators = []
                for model_name in top_models_info['Model']:
                    if model_name in self.models:
                        estimators.append((model_name, self.models[model_name]))
                
                if len(estimators) >= 2:
                    # 1. Voting Classifier
                    print("Training Voting Classifier...")
                    try:
                        voting_clf = VotingClassifier(
                            estimators=estimators,
                            voting='soft',
                            n_jobs=-1
                        )
                        
                        start_time = time.time()
                        voting_clf.fit(self.X_train_scaled, self.y_train)
                        training_time = time.time() - start_time
                        
                        y_pred = voting_clf.predict(self.X_test_scaled)
                        y_pred_proba = voting_clf.predict_proba(self.X_test_scaled)[:, 1]
                        
                        self.models['Voting_Classifier'] = voting_clf
                        
                        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                        metrics.update({
                            'Model': 'Voting_Classifier',
                            'Training_Time': training_time,
                            'Category': 'Ensemble'
                        })
                        
                        self.results.append(metrics)
                        print(f"✅ Voting Classifier: F1={metrics['F1_Score']:.4f}")
                        
                    except Exception as e:
                        print(f"❌ Voting Classifier failed: {str(e)}")
                    
                    # 2. Stacking Classifier
                    print("Training Stacking Classifier...")
                    try:
                        stacking_clf = StackingClassifier(
                            estimators=estimators,
                            final_estimator=LogisticRegression(),
                            cv=5,
                            n_jobs=-1
                        )
                        
                        start_time = time.time()
                        stacking_clf.fit(self.X_train_scaled, self.y_train)
                        training_time = time.time() - start_time
                        
                        y_pred = stacking_clf.predict(self.X_test_scaled)
                        y_pred_proba = stacking_clf.predict_proba(self.X_test_scaled)[:, 1]
                        
                        self.models['Stacking_Classifier'] = stacking_clf
                        
                        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                        metrics.update({
                            'Model': 'Stacking_Classifier',
                            'Training_Time': training_time,
                            'Category': 'Ensemble'
                        })
                        
                        self.results.append(metrics)
                        print(f"✅ Stacking Classifier: F1={metrics['F1_Score']:.4f}")
                        
                    except Exception as e:
                        print(f"❌ Stacking Classifier failed: {str(e)}")
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'AUC_ROC': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 0,
            'Confusion_Matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def analyze_results(self):
        """Analyze and compare all model results"""
        if not self.results:
            print("No results to analyze")
            return self
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('F1_Score', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(results_df[['Model', 'Category', 'Accuracy', 'Precision', 
                         'Recall', 'F1_Score', 'AUC_ROC', 'Training_Time']].to_string())
        
        # Identify best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models.get(self.best_model_name)
        self.best_score = results_df.iloc[0]['F1_Score']
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1 Score: {self.best_score:.4f}")
        print(f"   AUC ROC: {results_df.iloc[0]['AUC_ROC']:.4f}")
        
        return self
    
    def save_models_and_results(self, output_dir='models'):
        """Save all trained models and results"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'traditional'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'deep_learning'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ensemble'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'best_model'), exist_ok=True)
        
        # 1. Save all individual models
        for model_name, model in self.models.items():
            category = 'traditional'
            if any(x in model_name for x in ['XGBoost', 'LightGBM', 'CatBoost']):
                category = 'gradient_boosting'
            elif any(x in model_name for x in ['Neural', 'TabNet']):
                category = 'deep_learning'
            elif any(x in model_name for x in ['Voting', 'Stacking']):
                category = 'ensemble'
            
            model_path = os.path.join(output_dir, category, f'{model_name}.pkl')
            joblib.dump(model, model_path)
        
        # 2. Save best model with metadata
        if self.best_model:
            best_model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.X_train.columns.tolist(),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance': pd.DataFrame(self.results).to_dict('records')
            }
            
            best_model_path = os.path.join(output_dir, 'best_model', 'scalping_detector.pkl')
            joblib.dump(best_model_data, best_model_path)
            
            print(f"\n💾 Best model saved: {best_model_path}")
        
        # 3. Save results CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        # 4. Save detailed report
        report = {
            'summary': {
                'total_models_trained': len(self.models),
                'best_model': self.best_model_name,
                'best_f1_score': float(self.best_score),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_shape': self.X_train.shape
            },
            'model_performance': results_df.to_dict('records')
        }
        
        with open(os.path.join(output_dir, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"📊 Results saved to: {output_dir}/")
        
        return self
    
    def create_visualizations(self, output_dir='results'):
        """Create visualizations of model performance"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            return self
        
        results_df = pd.DataFrame(self.results)
        
        # 1. Bar chart of F1 scores
        plt.figure(figsize=(14, 8))
        results_sorted = results_df.sort_values('F1_Score', ascending=True)
        
        colors = []
        for category in results_sorted['Category']:
            if category == 'Traditional_ML':
                colors.append('skyblue')
            elif category == 'Gradient_Boosting':
                colors.append('lightgreen')
            elif category == 'Deep_Learning':
                colors.append('orange')
            elif category == 'Ensemble':
                colors.append('purple')
            else:
                colors.append('gray')
        
        bars = plt.barh(results_sorted['Model'], results_sorted['F1_Score'], color=colors)
        
        # Highlight best model
        best_idx = results_sorted['F1_Score'].idxmax()
        bars[best_idx].set_color('red')
        
        plt.xlabel('F1 Score')
        plt.title('Model Performance Comparison (F1 Score)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'f1_scores_comparison.png'), dpi=300)
        
        # 2. Radar chart for top 5 models
        top_5 = results_df.nlargest(5, 'F1_Score')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), 
                                subplot_kw=dict(polar=True) if len(metrics) > 2 else None)
        
        for idx, (_, row) in enumerate(top_5.iterrows()):
            if idx < 6:  # Limit to 6 subplots
                ax = axes[idx//3, idx%3]
                
                values = [row[metric] for metric in metrics]
                angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
                ax.set_title(f"{row['Model']} (F1: {row['F1_Score']:.3f})", size=14)
                ax.grid(True)
        
        plt.suptitle('Top 5 Models - Radar Chart Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300)
        
        # 3. Training time comparison
        plt.figure(figsize=(12, 6))
        results_sorted_time = results_df.sort_values('Training_Time', ascending=True)
        plt.barh(results_sorted_time['Model'], results_sorted_time['Training_Time'])
        plt.xlabel('Training Time (seconds)')
        plt.title('Model Training Time Comparison')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300)
        
        plt.show()
        print(f"📈 Visualizations saved to: {output_dir}/")
        
        return self
    
    def run_hyperparameter_tuning(self, model_name='XGBoost'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\n🔧 Performing Hyperparameter Tuning for {model_name}")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return self
        
        from sklearn.model_selection import RandomizedSearchCV
        
        if model_name == 'XGBoost':
            param_dist = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }
            
            model = xgb.XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        elif model_name == 'Random_Forest':
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
            
            model = RandomForestClassifier(random_state=42)
        
        # Randomized search
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='f1',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Tuning {model_name}...")
        random_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"✅ Best parameters: {random_search.best_params_}")
        print(f"✅ Best F1 score: {random_search.best_score_:.4f}")
        
        # Update model with tuned parameters
        self.models[f'{model_name}_Tuned'] = random_search.best_estimator_
        
        return self

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer('data/processed/engineered_data.csv')
    
    # Load and split data
    trainer.load_and_split_data(test_size=0.2)
    
    # Train all model categories
    trainer.train_traditional_models()\
          .train_gradient_boosting_models()\
          .train_deep_learning_models()\
          .train_ensemble_models()\
          .analyze_results()\
          .save_models_and_results('models/trained_models')\
          .create_visualizations('results/visualizations')
    
    # Optional: Hyperparameter tuning for best model
    # trainer.run_hyperparameter_tuning('XGBoost')
    
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"Total models trained: {len(trainer.models)}")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Best F1 score: {trainer.best_score:.4f}")