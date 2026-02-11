# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import create_directory, load_config, print_bold
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from scripts.train_all_models import ModelTrainer
from scripts.create_final_model import FinalModelSelector
from datetime import datetime

def main():
    """Main execution pipeline"""
    print_bold("\n" + "="*60)
    print_bold("🚀 REXELL AI - SCALPING DETECTION PIPELINE")
    print_bold("="*60)
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    directories = [
        'data/processed',
        'models/traditional',
        'models/boosting',
        'models/deep_learning',
        'models/ensemble',
        'models/best_model',
        'results/visualizations',
        'logs'
    ]
    
    for directory in directories:
        create_directory(directory)
    
    print("📁 Directories created successfully")
    
    # Step 1: Data Preprocessing
    print_bold("\n📊 STEP 1: DATA PREPROCESSING")
    preprocessor = DataPreprocessor(config['data']['raw_data_path'])
    df_cleaned = preprocessor.load_data()\
                             .handle_missing_values()\
                             .detect_outliers()\
                             .df
    
    # Save cleaned data
    df_cleaned.to_csv(config['data']['processed_data_path'], index=False)
    print(f"✅ Cleaned data saved: {config['data']['processed_data_path']}")
    
    # Step 2: Feature Engineering
    print_bold("\n🔧 STEP 2: FEATURE ENGINEERING")
    engineer = FeatureEngineer(df_cleaned)
    df_engineered, features = (engineer
                              .create_time_features()
                              .create_price_features()
                              .create_user_behavior_features()
                              .create_transaction_features()
                              .create_aggregate_features()
                              .create_risk_features()
                              .get_engineered_data())
    
    df_engineered.to_csv(config['data']['engineered_data_path'], index=False)
    print(f"✅ Engineered data saved with {len(features)} features")
    
    # Step 3: Model Training
    print_bold("\n🤖 STEP 3: MODEL TRAINING")
    trainer = ModelTrainer(config['data']['engineered_data_path'])
    
    trainer.load_and_split_data(
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    
    # Train models based on config
    if config['models']['traditional']:
        trainer.train_traditional_models()
    
    if config['models']['boosting']:
        trainer.train_gradient_boosting_models()
    
    if config['models']['deep_learning']:
        trainer.train_deep_learning_models()
    
    if config['models']['ensemble']:
        trainer.train_ensemble_models()
    
    # Analyze and save results
    trainer.analyze_results()\
           .save_models_and_results(config['output']['models_dir'])\
           .create_visualizations(config['output']['visualizations_dir'])
    
    # Step 4: Final Model Selection
    print_bold("\n🏆 STEP 4: FINAL MODEL SELECTION")
    
    # Get the best model from training
    best_model_name = trainer.best_model_name
    best_model = trainer.best_model
    
    if best_model:
        print(f"Selected best model: {best_model_name}")
        
        # Prepare data for final package
        X = df_engineered[features]
        y = df_engineered['scalping_label']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state'],
            stratify=y
        )
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create final package
        selector = FinalModelSelector()
        final_package = selector.create_final_model_package(
            model=best_model,
            scaler=scaler,
            feature_names=features,
            X_test=X_test,
            y_test=y_test,
            output_path='models/best_model/scalping_detection_final.pkl'
        )
    
    print_bold("\n" + "="*60)
    print_bold("🎉 PIPELINE EXECUTION COMPLETE!")
    print_bold("="*60)
    
    # Summary
    print("\n📋 SUMMARY:")
    print(f"• Data processed: {df_cleaned.shape[0]} transactions")
    print(f"• Features created: {len(features)}")
    print(f"• Models trained: {len(trainer.models)}")
    print(f"• Best model: {best_model_name}")
    print(f"• Final model saved: models/best_model/scalping_detection_final.pkl")
    
    # Next steps
    print_bold("\n📌 NEXT STEPS:")
    print("1. Check results/visualizations/ for performance charts")
    print("2. Review models/model_comparison.csv")
    print("3. Test the model with sample data")
    print("4. Deliver the .pkl file to your teammate")

if __name__ == "__main__":
    # Record start time
    start_time = datetime.now()
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        from utils.helpers import timer
        timer(start_time, "Complete pipeline")