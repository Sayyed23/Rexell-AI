# scripts/train_all_models.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
from sklearn.calibration import CalibratedClassifierCV

import joblib
import json
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ─── PyTorch Custom Neural Network ───
class ScalpingDetectorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ─── Sklearn-compatible wrapper for the PyTorch model ───
class PyTorchClassifierWrapper:
    """Wraps the PyTorch model so it can be stored and evaluated uniformly."""

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            t = torch.FloatTensor(np.asarray(X)).to(self.device)
            proba = self.net(t).cpu().numpy()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            t = torch.FloatTensor(np.asarray(X)).to(self.device)
            p1 = self.net(t).cpu().numpy()
        return np.column_stack([1 - p1, p1])


class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_scaled = self.X_test_scaled = None
        self.scaler = None
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    # ──────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────
    def load_and_split_data(self, test_size=0.2, random_state=42):
        print("📊 Loading data...")
        df = pd.read_csv(self.data_path)

        X = df.drop(['scalping_label', 'transaction_hash', 'wallet_address',
                     'timestamp', 'fraud_label'], axis=1, errors='ignore')
        y = df['scalping_label']

        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✅ Data loaded: {X.shape}")
        print(f"   Training: {self.X_train.shape}, Testing: {self.X_test.shape}")
        print(f"   Target dist (train): {dict(self.y_train.value_counts())}")
        return self

    # ──────────────────────────────────────────────
    #  METRICS HELPER
    # ──────────────────────────────────────────────
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'AUC_ROC': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 0,
            'Confusion_Matrix': confusion_matrix(y_true, y_pred).tolist()
        }

    def _log_model_result(self, name, category, metrics, training_time, cv_scores=None):
        metrics.update({
            'Model': name,
            'Training_Time': round(training_time, 4),
            'Category': category
        })
        if cv_scores is not None:
            metrics['CV_Mean_F1'] = round(float(np.mean(cv_scores)), 4)
            metrics['CV_Std_F1'] = round(float(np.std(cv_scores)), 4)

        self.results.append(metrics)

        print(f"\n  ✅ {name}")
        print(f"     Accuracy : {metrics['Accuracy']:.4f}")
        print(f"     Precision: {metrics['Precision']:.4f}")
        print(f"     Recall   : {metrics['Recall']:.4f}")
        print(f"     F1 Score : {metrics['F1_Score']:.4f}")
        print(f"     AUC-ROC  : {metrics['AUC_ROC']:.4f}")
        print(f"     Time     : {training_time:.2f}s")
        if cv_scores is not None:
            print(f"     CV F1    : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"     Confusion Matrix: {metrics['Confusion_Matrix']}")

    # ──────────────────────────────────────────────
    #  1) TRADITIONAL ML MODELS
    # ──────────────────────────────────────────────
    def train_traditional_models(self):
        print("\n" + "=" * 60)
        print("🔵 TRAINING TRADITIONAL ML MODELS")
        print("=" * 60)

        models_dict = {
            'Logistic_Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'Decision_Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
            'Gradient_Boosting_SKL': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Extra_Trees': ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB(),
            'Bagging': BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (name, model) in enumerate(models_dict.items(), 1):
            print(f"\n  [{i}/{len(models_dict)}] Training {name}...")
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train,
                                           cv=cv, scoring='f1', n_jobs=-1)
                print(f"     5-fold CV F1 scores: {[round(s, 4) for s in cv_scores]}")

                # Full training
                start = time.time()
                model.fit(self.X_train_scaled, self.y_train)
                train_time = time.time() - start

                y_pred = model.predict(self.X_test_scaled)
                y_proba = model.predict_proba(self.X_test_scaled)[:, 1]

                self.models[name] = model
                metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
                self._log_model_result(name, 'Traditional_ML', metrics, train_time, cv_scores)

            except Exception as e:
                print(f"  ❌ {name} failed: {e}")

        print(f"\n  📋 Traditional models completed: {sum(1 for r in self.results if r['Category']=='Traditional_ML')}/{len(models_dict)}")
        return self

    # ──────────────────────────────────────────────
    #  2) GRADIENT BOOSTING MODELS
    # ──────────────────────────────────────────────
    def train_gradient_boosting_models(self):
        print("\n" + "=" * 60)
        print("🟢 TRAINING GRADIENT BOOSTING MODELS")
        print("=" * 60)

        pos_weight = len(self.y_train[self.y_train == 0]) / max(len(self.y_train[self.y_train == 1]), 1)

        boosting_models = {
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss', scale_pos_weight=pos_weight,
                    verbosity=1
                ),
                'use_scaled': True
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=31,
                    max_depth=-1, random_state=42, class_weight='balanced',
                    n_jobs=-1, verbose=1
                ),
                'use_scaled': True
            },
            'CatBoost': {
                'model': cb.CatBoostClassifier(
                    iterations=200, learning_rate=0.05, depth=6,
                    random_seed=42, verbose=20,
                    auto_class_weights='Balanced'
                ),
                'use_scaled': False  # CatBoost handles raw data
            }
        }

        for i, (name, info) in enumerate(boosting_models.items(), 1):
            print(f"\n  [{i}/{len(boosting_models)}] Training {name}...")
            model = info['model']
            use_scaled = info['use_scaled']
            try:
                X_tr = self.X_train_scaled if use_scaled else self.X_train
                X_te = self.X_test_scaled if use_scaled else self.X_test

                start = time.time()
                if name == 'XGBoost':
                    model.fit(X_tr, self.y_train,
                              eval_set=[(X_te, self.y_test)],
                              verbose=50)
                elif name == 'LightGBM':
                    model.fit(X_tr, self.y_train,
                              eval_set=[(X_te, self.y_test)],
                              callbacks=[lgb.log_evaluation(50)])
                elif name == 'CatBoost':
                    model.fit(X_tr, self.y_train,
                              eval_set=(X_te, self.y_test))
                else:
                    model.fit(X_tr, self.y_train)

                train_time = time.time() - start

                y_pred = model.predict(X_te)
                y_proba = model.predict_proba(X_te)[:, 1]

                self.models[name] = model
                metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
                self._log_model_result(name, 'Gradient_Boosting', metrics, train_time)

            except Exception as e:
                print(f"  ❌ {name} failed: {e}")

        print(f"\n  📋 Boosting models completed: {sum(1 for r in self.results if r['Category']=='Gradient_Boosting')}/{len(boosting_models)}")
        return self

    # ──────────────────────────────────────────────
    #  3) DEEP LEARNING MODELS
    # ──────────────────────────────────────────────
    def train_deep_learning_models(self, epochs=100, batch_size=64, lr=0.001, patience=15):
        print("\n" + "=" * 60)
        print("🟠 TRAINING DEEP LEARNING MODELS")
        print("=" * 60)

        # --- 3a. MLP (sklearn) with verbose epoch logging ---
        print("\n  [1/3] Training Neural Network (MLP)...")
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                alpha=0.0001, batch_size=32, learning_rate='adaptive',
                max_iter=epochs, random_state=42, early_stopping=True,
                validation_fraction=0.15, verbose=True
            )
            start = time.time()
            mlp.fit(self.X_train_scaled, self.y_train)
            train_time = time.time() - start

            print(f"\n     MLP converged after {mlp.n_iter_} iterations")
            print(f"     Final training loss: {mlp.loss_curve_[-1]:.6f}")

            y_pred = mlp.predict(self.X_test_scaled)
            y_proba = mlp.predict_proba(self.X_test_scaled)[:, 1]

            self.models['Neural_Network_MLP'] = mlp
            metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
            self._log_model_result('Neural_Network_MLP', 'Deep_Learning', metrics, train_time)

        except Exception as e:
            print(f"  ❌ MLP failed: {e}")

        # --- 3b. TabNet with epoch logging ---
        print("\n  [2/3] Training TabNet...")
        try:
            tabnet = TabNetClassifier(
                n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='entmax',
                scheduler_params={"step_size": 10, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=1, seed=42
            )
            start = time.time()
            tabnet.fit(
                self.X_train_scaled, self.y_train.values,
                eval_set=[(self.X_test_scaled, self.y_test.values)],
                eval_metric=['auc', 'accuracy'],
                max_epochs=min(epochs, 100), patience=patience,
                batch_size=1024, virtual_batch_size=128
            )
            train_time = time.time() - start

            print(f"\n     TabNet best epoch: {tabnet.best_epoch if hasattr(tabnet, 'best_epoch') else 'N/A'}")

            y_pred = tabnet.predict(self.X_test_scaled)
            y_proba = tabnet.predict_proba(self.X_test_scaled)[:, 1]

            self.models['TabNet'] = tabnet
            metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
            self._log_model_result('TabNet', 'Deep_Learning', metrics, train_time)

        except Exception as e:
            print(f"  ❌ TabNet failed: {e}")

        # --- 3c. Custom PyTorch NN with full epoch-by-epoch logging ---
        print("\n  [3/3] Training PyTorch Neural Network...")
        try:
            self._train_pytorch_nn(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)
        except Exception as e:
            print(f"  ❌ PyTorch NN failed: {e}")

        dl_count = sum(1 for r in self.results if r['Category'] == 'Deep_Learning')
        print(f"\n  📋 Deep learning models completed: {dl_count}/3")
        return self

    def _train_pytorch_nn(self, epochs=100, batch_size=64, lr=0.001, patience=15):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"     Device: {device}")

        X_tr = torch.FloatTensor(self.X_train_scaled).to(device)
        y_tr = torch.FloatTensor(self.y_train.values).to(device)
        X_te = torch.FloatTensor(self.X_test_scaled).to(device)
        y_te = torch.FloatTensor(self.y_test.values).to(device)

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = ScalpingDetectorNet(X_tr.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCELoss()

        best_val_f1 = 0
        no_improve = 0
        best_state = None

        print(f"\n     {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>6} | {'LR':>10}")
        print("     " + "-" * 75)

        start = time.time()
        for epoch in range(1, epochs + 1):
            # --- Train ---
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                preds = (out >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                val_out = model(X_te)
                val_loss = criterion(val_out, y_te).item()
                val_preds = (val_out >= 0.5).float().cpu().numpy()
                val_acc = accuracy_score(self.y_test, val_preds)
                val_f1 = f1_score(self.y_test, val_preds, zero_division=0)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            # Log every epoch
            print(f"     {epoch:>6} | {train_loss:>10.6f} | {train_acc:>8.4f}  | {val_loss:>8.5f} | {val_acc:>6.4f}  | {val_f1:>5.4f} | {current_lr:>10.6f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n     ⚠ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        train_time = time.time() - start

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        print(f"\n     Best validation F1: {best_val_f1:.4f}")

        wrapper = PyTorchClassifierWrapper(model, device)
        y_pred = wrapper.predict(self.X_test_scaled)
        y_proba = wrapper.predict_proba(self.X_test_scaled)[:, 1]

        self.models['PyTorch_NN'] = wrapper
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        self._log_model_result('PyTorch_NN', 'Deep_Learning', metrics, train_time)

    # ──────────────────────────────────────────────
    #  4) ENSEMBLE MODELS
    # ──────────────────────────────────────────────
    def train_ensemble_models(self):
        print("\n" + "=" * 60)
        print("🟣 TRAINING ENSEMBLE MODELS")
        print("=" * 60)

        # Filter only sklearn-compatible models (exclude TabNet, PyTorch wrappers)
        sklearn_models = {}
        skip_names = {'TabNet', 'PyTorch_NN', 'Neural_Network_MLP'}
        for name, mdl in self.models.items():
            if name not in skip_names and hasattr(mdl, 'get_params'):
                sklearn_models[name] = mdl

        if len(sklearn_models) < 2:
            print("  ⚠ Not enough sklearn-compatible models for ensemble (need ≥2)")
            return self

        results_df = pd.DataFrame(self.results)
        sklearn_results = results_df[results_df['Model'].isin(sklearn_models.keys())]
        if sklearn_results.empty:
            print("  ⚠ No results for sklearn models")
            return self

        top_models = sklearn_results.nlargest(min(3, len(sklearn_results)), 'F1_Score')['Model'].tolist()
        estimators = [(n, sklearn_models[n]) for n in top_models]
        print(f"  Top models for ensemble: {top_models}")

        # Voting Classifier
        print("\n  [1/2] Training Voting Classifier...")
        try:
            voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            start = time.time()
            voting.fit(self.X_train_scaled, self.y_train)
            t = time.time() - start

            y_pred = voting.predict(self.X_test_scaled)
            y_proba = voting.predict_proba(self.X_test_scaled)[:, 1]
            self.models['Voting_Classifier'] = voting
            self._log_model_result('Voting_Classifier', 'Ensemble',
                                   self._calculate_metrics(self.y_test, y_pred, y_proba), t)
        except Exception as e:
            print(f"  ❌ Voting Classifier failed: {e}")

        # Stacking Classifier
        print("\n  [2/2] Training Stacking Classifier...")
        try:
            stacking = StackingClassifier(
                estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
            start = time.time()
            stacking.fit(self.X_train_scaled, self.y_train)
            t = time.time() - start

            y_pred = stacking.predict(self.X_test_scaled)
            y_proba = stacking.predict_proba(self.X_test_scaled)[:, 1]
            self.models['Stacking_Classifier'] = stacking
            self._log_model_result('Stacking_Classifier', 'Ensemble',
                                   self._calculate_metrics(self.y_test, y_pred, y_proba), t)
        except Exception as e:
            print(f"  ❌ Stacking Classifier failed: {e}")

        return self

    # ──────────────────────────────────────────────
    #  ANALYSIS & RESULTS
    # ──────────────────────────────────────────────
    def analyze_results(self):
        if not self.results:
            print("No results to analyze")
            return self

        df = pd.DataFrame(self.results).sort_values('F1_Score', ascending=False)

        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 60)

        display_cols = ['Model', 'Category', 'Accuracy', 'Precision', 'Recall',
                        'F1_Score', 'AUC_ROC', 'Training_Time']
        if 'CV_Mean_F1' in df.columns:
            display_cols += ['CV_Mean_F1', 'CV_Std_F1']
        print(df[display_cols].to_string(index=False))

        self.best_model_name = df.iloc[0]['Model']
        self.best_model = self.models.get(self.best_model_name)
        self.best_score = df.iloc[0]['F1_Score']

        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1 Score: {self.best_score:.4f}")
        print(f"   AUC ROC : {df.iloc[0]['AUC_ROC']:.4f}")
        return self

    def save_models_and_results(self, output_dir='models'):
        os.makedirs(output_dir, exist_ok=True)
        for sub in ['traditional', 'gradient_boosting', 'deep_learning', 'ensemble', 'best_model']:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        for name, model in self.models.items():
            cat = 'traditional'
            if any(x in name for x in ['XGBoost', 'LightGBM', 'CatBoost']):
                cat = 'gradient_boosting'
            elif any(x in name for x in ['Neural', 'TabNet', 'PyTorch']):
                cat = 'deep_learning'
            elif any(x in name for x in ['Voting', 'Stacking']):
                cat = 'ensemble'
            path = os.path.join(output_dir, cat, f'{name}.pkl')
            joblib.dump(model, path)

        if self.best_model:
            data = {
                'model': self.best_model, 'scaler': self.scaler,
                'feature_names': self.X_train.columns.tolist(),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'performance': self.results
            }
            joblib.dump(data, os.path.join(output_dir, 'best_model', 'scalping_detector.pkl'))

        pd.DataFrame(self.results).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

        report = {
            'summary': {
                'total_models_trained': len(self.models),
                'best_model': self.best_model_name or 'None',
                'best_f1_score': float(self.best_score),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_rows': int(self.X_train.shape[0]),
                'dataset_features': int(self.X_train.shape[1])
            }
        }
        with open(os.path.join(output_dir, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=4, default=str)

        print(f"\n💾 All models & results saved to: {output_dir}/")
        return self

    # ──────────────────────────────────────────────
    #  VISUALIZATIONS
    # ──────────────────────────────────────────────
    def create_visualizations(self, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)
        if not self.results:
            return self

        df = pd.DataFrame(self.results)

        # 1. F1 Score bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        sorted_df = df.sort_values('F1_Score', ascending=True).reset_index(drop=True)
        color_map = {'Traditional_ML': 'skyblue', 'Gradient_Boosting': 'lightgreen',
                     'Deep_Learning': 'orange', 'Ensemble': 'purple'}
        colors = [color_map.get(c, 'gray') for c in sorted_df['Category']]
        bars = ax.barh(sorted_df['Model'], sorted_df['F1_Score'], color=colors)

        best_pos = sorted_df['F1_Score'].idxmax()
        bars[best_pos].set_color('red')
        bars[best_pos].set_edgecolor('darkred')
        bars[best_pos].set_linewidth(2)

        ax.set_xlabel('F1 Score')
        ax.set_title('Model Performance Comparison (F1 Score)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'f1_scores_comparison.png'), dpi=300)
        plt.close(fig)

        # 2. Grouped metrics bar chart
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
        top5 = df.nlargest(min(5, len(df)), 'F1_Score')
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(top5))
        w = 0.15
        for i, m in enumerate(metric_cols):
            ax.bar(x + i * w, top5[m], w, label=m)
        ax.set_xticks(x + w * 2)
        ax.set_xticklabels(top5['Model'], rotation=30, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Top Models — All Metrics Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
        plt.close(fig)

        # 3. Training time
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_time = df.sort_values('Training_Time', ascending=True)
        ax.barh(sorted_time['Model'], sorted_time['Training_Time'], color='steelblue')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300)
        plt.close(fig)

        # 4. Category-wise box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = df['Category'].unique()
        data_per_cat = [df[df['Category'] == c]['F1_Score'].values for c in categories]
        ax.boxplot(data_per_cat, labels=categories)
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Distribution by Model Category')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=300)
        plt.close(fig)

        print(f"📈 Visualizations saved to: {output_dir}/")
        return self

    # ──────────────────────────────────────────────
    #  HYPERPARAMETER TUNING
    # ──────────────────────────────────────────────
    def run_hyperparameter_tuning(self, model_name='XGBoost'):
        from sklearn.model_selection import RandomizedSearchCV
        print(f"\n🔧 Hyperparameter Tuning for {model_name}")

        if model_name not in self.models:
            print(f"  Model {model_name} not found")
            return self

        param_dists = {
            'XGBoost': ({'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
                         'max_depth': [3, 6, 9], 'subsample': [0.6, 0.8, 1.0],
                         'colsample_bytree': [0.6, 0.8, 1.0]},
                        xgb.XGBClassifier(random_state=42, eval_metric='logloss')),
            'Random_Forest': ({'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15, None],
                               'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                              RandomForestClassifier(random_state=42)),
        }

        if model_name not in param_dists:
            print(f"  No tuning config for {model_name}")
            return self

        params, base = param_dists[model_name]
        search = RandomizedSearchCV(base, params, n_iter=20, cv=3, scoring='f1',
                                    random_state=42, n_jobs=-1, verbose=1)
        search.fit(self.X_train_scaled, self.y_train)
        print(f"  ✅ Best params: {search.best_params_}")
        print(f"  ✅ Best CV F1 : {search.best_score_:.4f}")
        self.models[f'{model_name}_Tuned'] = search.best_estimator_
        return self


# ═══════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE MODEL TRAINING")
    print("=" * 60)

    trainer = ModelTrainer('data/processed/engineered_data.csv')
    trainer.load_and_split_data(test_size=0.2)

    trainer.train_traditional_models()
    trainer.train_gradient_boosting_models()
    trainer.train_deep_learning_models(epochs=100, batch_size=64, lr=0.001, patience=15)
    trainer.train_ensemble_models()

    trainer.analyze_results()
    trainer.save_models_and_results('models/trained_models')
    trainer.create_visualizations('results/visualizations')

    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total models trained: {len(trainer.models)}")
    if trainer.best_model_name:
        print(f"Best model: {trainer.best_model_name}")
        print(f"Best F1 score: {trainer.best_score:.4f}")