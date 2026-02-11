# scripts/create_final_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FinalModelSelector:
    def __init__(self, models_dir='models/trained_models'):
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        
    def load_all_models(self):
        """Load all trained models"""
        import os
        
        for category in ['traditional', 'gradient_boosting', 'deep_learning', 'ensemble']:
            category_path = os.path.join(self.models_dir, category)
            if os.path.exists(category_path):
                for file in os.listdir(category_path):
                    if file.endswith('.pkl'):
                        model_name = file.replace('.pkl', '')
                        model_path = os.path.join(category_path, file)
                        
                        try:
                            model = joblib.load(model_path)
                            self.models[model_name] = model
                            print(f"✅ Loaded: {model_name}")
                        except:
                            print(f"❌ Failed to load: {model_name}")
        
        print(f"\n📦 Total models loaded: {len(self.models)}")
        return self
    
    def select_best_model(self, X, y, cv_folds=5):
        """Select best model using cross-validation"""
        print("\n🔍 Selecting best model via cross-validation...")
        
        best_score = 0
        best_model_name = None
        best_model = None
        
        for name, model in self.models.items():
            try:
                # Cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                self.results[name] = {
                    'mean_f1': mean_score,
                    'std_f1': std_score,
                    'cv_scores': scores.tolist()
                }
                
                print(f"{name:<25} F1: {mean_score:.4f} (±{std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    best_model = model
                    
            except Exception as e:
                print(f"{name:<25} Failed: {str(e)}")
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Cross-validated F1: {best_score:.4f}")
        
        return best_model_name, best_model
    
    def create_final_model_package(self, model, scaler, feature_names, 
                                  X_test, y_test, output_path='final_model_package.pkl'):
        """Create final model package for delivery"""
        
        # Test final predictions
        if hasattr(scaler, 'transform'):
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Create model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'metrics': metrics,
            'model_type': type(model).__name__,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_performance': {
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        }
        
        # Save package
        joblib.dump(model_package, output_path)
        
        print(f"\n🎁 Final model package saved: {output_path}")
        print(f"📊 Test Performance:")
        for metric, value in metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        # Create performance visualization
        self._create_performance_visualization(y_test, y_pred, y_pred_proba, 
                                              metrics, output_path.replace('.pkl', '_performance.png'))
        
        return model_package
    
    def _create_performance_visualization(self, y_test, y_pred, y_pred_proba, 
                                         metrics, output_path):
        """Create performance visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Metrics Bar Chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        axes[0, 1].bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple', 'orange'])
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # 3. ROC Curve
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[1, 0].plot(fpr, tpr, label=f'AUC = {metrics.get("roc_auc", 0):.3f}', linewidth=2)
            axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            top_features = np.argsort(feature_importance)[-10:]  # Top 10 features
            
            axes[1, 1].barh(range(len(top_features)), feature_importance[top_features])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in top_features])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.suptitle('Final Model Performance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance visualization saved: {output_path}")

# Main execution
if __name__ == "__main__":
    # Load data
    from src.feature_engineering import FeatureEngineer
    from src.data_preprocessing import DataPreprocessor
    
    print("📂 Loading and preparing data...")
    
    # Preprocess
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
    
    # Prepare X and y
    X = df_engineered[features]
    y = df_engineered['scalping_label']
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data prepared")
    print(f"   Training: {X_train.shape}, Testing: {X_test.shape}")
    
    # Load models and select best
    selector = FinalModelSelector('models/trained_models')
    selector.load_all_models()
    
    best_name, best_model = selector.select_best_model(X_train_scaled, y_train)
    
    # Create final model package
    final_package = selector.create_final_model_package(
        model=best_model,
        scaler=scaler,
        feature_names=features,
        X_test=X_test,
        y_test=y_test,
        output_path='final_delivery/scalping_detection_model.pkl'
    )
    
    print("\n" + "="*60)
    print("🎉 FINAL MODEL READY FOR DELIVERY!")
    print("="*60)
    print("Deliver these files to your teammate:")
    print("1. final_delivery/scalping_detection_model.pkl")
    print("2. final_delivery/scalping_detection_model_performance.png")
    print("3. models/trained_models/model_comparison.csv")
    print("4. results/visualizations/ (all charts)")