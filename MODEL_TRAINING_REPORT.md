# Rexell AI: Scalping Detection Model Training Report

## 1. Module Description & Overall Implementation Plan

The **Rexell AI Scalping Detection Pipeline** is a state-of-the-art machine learning ecosystem designed to identify and mitigate ticket scalping on digital platforms. The pipeline is built with a modular architecture to ensure scalability, interpretability, and high performance.

### Implementation Architecture
- **Data Ingestion & Cleaning**: Processes raw transaction logs, handles missing data, and implements robust outlier detection using the IQR (Interquartile Range) method.
- **Advanced Feature Engineering**: Transforms raw data into high-dimensional behavioral signals, including wallet transaction density, price intensity metrics, and temporal patterns (e.g., bot-like transaction timing).
- **Multi-Category Training Engine**: A unified framework that trains and evaluates:
    - **Traditional ML**: Baseline robustness (Random Forest, SVM).
    - **Gradient Boosting**: Optimized for tabular accuracy (XGBoost, CatBoost, LightGBM).
    - **Tabular Deep Learning**: Capturing non-linear interactions (TabNet, PyTorch NN).
- **Automated Selection & Deployment**: Identifies the best-performing model based on F1-score and generates a production-ready model package including pre-processing transforms.

---

## 2. Algorithms, Models & Hardware Specifications

### Algorithm Deep Dive
- **XGBoost (Extreme Gradient Boosting)**: Current top performer. Utilizes parallel tree boosting to achieve high precision in imbalanced classification tasks.
- **CatBoost**: Leverages innovative handling of categorical features and provides stable performance with minimal hyperparameter tuning.
- **Custom PyTorch Neural Network**: A deep residual network architecture optimized for high-dimensional ticket transaction data.

### Hardware Infrastructure
The training was executed on a dedicated GPU-accelerated environment:
- **GPU**: NVIDIA GeForce GTX 1650 (4GB GDDR5).
- **Core Platform**: CUDA 13.0 / cuDNN 9.x.
- **Acceleration**: Boosting models utilize `gpu_hist` for rapid iteration; Deep Learning models leverage `torch.cuda` for parallel weight updates.
- **Runtime Environment**: Python 3.11.14 (Strict isolation via `.venv`).

---

## 3. Dataset Confirmation

The model training utilizes a comprehensive behavioral dataset derived from ticket secondary markets.

| Metric | Value |
| :--- | :--- |
| **Total Rows (Pruned)** | 9,600 transactions |
| **Feature Dimensionality** | 19,653 engineered features |
| **Target Variable** | `scalping_label` (Binary) |
| **Train/Test Split** | 80/20 Stratified |
| **Primary Features** | `markup_pct`, `ticket_count`, `time_between_tx`, `ip_occurrence_freq` |

---

## 4. Module-Based Sample Results

The latest training run confirms the effectiveness of the Gradient Boosting suite, particularly in maximizing the F1-score for rare scalping events.

### Performance Summary
| Model Name | Accuracy | Precision | Recall | **F1-Score** | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (GPU)** | **98.50%** | 0.8817 | **0.9879** | **0.9318** | 49.0s |
| **LightGBM** | 98.41% | 0.8808 | 0.9799 | 0.9277 | 15.9s |
| **Neural Network (MLP)** | 94.54% | **0.9758** | 0.4859 | 0.6487 | 290.2s |

### Conclusion
The **XGBoost** model has been selected as the primary detection engine due to its exceptional **98.7% Recall**, ensuring that almost all scalping attempts are successfully flagged while maintaining high operational accuracy.
