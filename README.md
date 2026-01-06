# Claims Fraud Detection System

A comprehensive fraud detection system for insurance claims using advanced machine learning techniques. This system demonstrates state-of-the-art approaches to fraud detection with proper evaluation metrics, cost-benefit analysis, and interactive visualization.

## ⚠️ IMPORTANT DISCLAIMER

**THIS SYSTEM IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

- This is NOT intended for actual fraud detection in production environments
- Results may be inaccurate and should not be used for investment or business decisions
- This system is designed for learning and demonstration purposes
- No warranty or guarantee of accuracy is provided
- Use at your own risk

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. **Train the models:**
   ```bash
   python scripts/train.py
   ```

2. **Launch the interactive demo:**
   ```bash
   streamlit run demo/app.py
   ```

3. **Access the demo** in your browser at `http://localhost:8501`

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── data/                     # Data generation and processing
│   │   └── synthetic_data.py     # Synthetic claims data generator
│   ├── features/                 # Feature engineering
│   │   └── feature_engineering.py # Feature preprocessing and selection
│   ├── models/                   # Machine learning models
│   │   ├── xgboost_model.py      # XGBoost fraud detector
│   │   ├── lightgbm_model.py     # LightGBM fraud detector
│   │   └── isolation_forest_model.py # Isolation Forest detector
│   ├── evaluation/               # Model evaluation
│   │   └── fraud_metrics.py      # Comprehensive evaluation metrics
│   └── utils/                    # Utilities
│       ├── config.py             # Configuration management
│       └── logging.py            # Logging utilities
├── configs/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   ├── data/                     # Data configurations
│   ├── model/                    # Model configurations
│   └── evaluation/               # Evaluation configurations
├── scripts/                       # Training and utility scripts
│   └── train.py                  # Main training pipeline
├── demo/                         # Interactive demo
│   └── app.py                   # Streamlit application
├── tests/                        # Unit tests
├── assets/                       # Generated assets and plots
├── models/                       # Saved trained models
├── logs/                         # Log files
├── data/                         # Data storage
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Features

### Advanced Machine Learning Models

- **XGBoost**: Gradient boosting with early stopping and feature importance
- **LightGBM**: Fast gradient boosting with categorical feature handling
- **Isolation Forest**: Unsupervised anomaly detection for fraud patterns

### Comprehensive Evaluation

- **Standard Metrics**: AUC, Precision, Recall, F1-Score, Accuracy
- **Fraud-Specific Metrics**: Precision@K, Recall@K for top-K analysis
- **Cost-Benefit Analysis**: Investigation costs vs. fraud prevention benefits
- **Calibration Metrics**: Brier score and calibration error
- **Threshold Optimization**: Youden's J, F1 optimization, Precision-Recall balance

### Feature Engineering

- **Categorical Encoding**: Target encoding and label encoding
- **Feature Scaling**: Standard scaling for numerical features
- **Feature Selection**: Statistical feature selection (F-test)
- **Derived Features**: Age groups, high-value flags, frequent claimant indicators

### Interactive Demo

- **Manual Claim Entry**: Input individual claims for analysis
- **Batch Analysis**: Analyze multiple claims simultaneously
- **Model Comparison**: Compare performance across different models
- **Visualization**: Interactive charts and graphs
- **Feature Importance**: Understand which features drive predictions

## Dataset Schema

The system generates synthetic claims data with the following features:

### Numerical Features
- `claim_amount`: Claim amount in USD (log-normal distribution)
- `customer_age`: Customer age (normal distribution, 18-80)
- `num_previous_claims`: Number of previous claims (Poisson distribution)
- `days_since_last_claim`: Days since last claim (exponential distribution)
- `policy_duration`: Policy duration in years (normal distribution)

### Categorical Features
- `claim_type`: Type of claim (Accident, Fire, Theft, Health, Property)
- `location`: Geographic location (Urban, Suburban, Rural)
- `customer_segment`: Customer tier (Premium, Standard, Basic)
- `claim_time`: Time of claim submission (Business_Hours, Evening, Night, Weekend)

### Derived Features
- `claim_amount_per_previous`: Claim amount normalized by previous claims
- `age_group`: Age categories (Young, Adult, Middle, Senior, Elderly)
- `high_value_claim`: Flag for high-value claims (top 10%)
- `frequent_claimant`: Flag for customers with 3+ previous claims
- `recent_claim`: Flag for claims within 30 days of previous claim

## Model Performance

The system evaluates models using multiple metrics:

### Classification Metrics
- **AUC**: Area Under the ROC Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)

### Fraud-Specific Metrics
- **Precision@K**: Fraction of top-K predictions that are actually fraud
- **Recall@K**: Fraction of all fraud cases captured in top-K predictions
- **Cost-Benefit Analysis**: ROI considering investigation costs and fraud losses

### Calibration Metrics
- **Brier Score**: Mean squared error of probability predictions
- **Calibration Error**: Average absolute difference between predicted and actual probabilities

## Configuration

The system uses YAML configuration files for easy customization:

### Main Configuration (`configs/config.yaml`)
- Data generation parameters
- Model hyperparameters
- Evaluation settings
- Path configurations

### Model-Specific Configurations
- XGBoost parameters (`configs/model/xgboost.yaml`)
- LightGBM parameters (`configs/model/lightgbm.yaml`)
- Isolation Forest parameters (`configs/model/isolation_forest.yaml`)

### Evaluation Configuration (`configs/evaluation/fraud_metrics.yaml`)
- Metrics to calculate
- K-values for Precision@K and Recall@K
- Cost-benefit analysis parameters
- Calibration settings

## Usage Examples

### Training Models
```python
from scripts.train import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline("configs/config.yaml")

# Run complete pipeline
results = pipeline.run_pipeline()
```

### Making Predictions
```python
from src.models.xgboost_model import XGBoostFraudDetector

# Load trained model
model = XGBoostFraudDetector({})
model.load_model("models/xgboost_model.pkl")

# Make prediction
claim_data = pd.DataFrame([{
    'claim_amount': 5000,
    'customer_age': 45,
    'num_previous_claims': 1,
    # ... other features
}])

prediction = model.predict(claim_data)
probability = model.predict_proba(claim_data)
```

### Evaluating Models
```python
from src.evaluation.fraud_metrics import FraudDetectionEvaluator

# Initialize evaluator
evaluator = FraudDetectionEvaluator({
    'metrics': ['auc', 'precision_at_k', 'recall_at_k'],
    'k_values': [10, 50, 100]
})

# Evaluate model
metrics = evaluator.evaluate(y_true, y_pred, y_proba)
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Monitoring and Logging

The system includes comprehensive logging:
- Training progress and metrics
- Model performance evaluation
- Error handling and debugging information
- Timestamped log files in the `logs/` directory

## Security and Compliance

- **Data Privacy**: Synthetic data only, no real customer information
- **Model Validation**: Comprehensive evaluation with multiple metrics
- **Audit Trail**: Complete logging of model training and evaluation
- **Reproducibility**: Deterministic random seeds and version control

## Contributing

This is a research and educational project. Contributions are welcome for:
- Additional model implementations
- New evaluation metrics
- Enhanced visualization features
- Documentation improvements

## References

- XGBoost: [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)
- LightGBM: [Ke et al., 2017](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- Isolation Forest: [Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- Fraud Detection: [Bolton & Hand, 2002](https://www.jstor.org/stable/3087254)

## License

This project is for educational and research purposes only. Please ensure compliance with your local regulations and institutional policies when using this system.

## Support

For questions or issues:
1. Check the documentation in this README
2. Review the configuration files
3. Check the log files for error messages
4. Ensure all dependencies are properly installed

---

**Remember: This system is for research and educational purposes only. Do not use for actual fraud detection without proper validation and compliance measures.**
# Claims-Fraud-Detection-System
