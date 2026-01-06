"""Streamlit demo for fraud detection system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.synthetic_data import SyntheticClaimsData, ClaimsDataConfig
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostFraudDetector
from src.models.lightgbm_model import LightGBMFraudDetector
from src.models.isolation_forest_model import IsolationForestFraudDetector
from src.evaluation.fraud_metrics import FraudDetectionEvaluator

# Page configuration
st.set_page_config(
    page_title="Claims Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
    .legitimate-claim {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛡️ Claims Fraud Detection System</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ DISCLAIMER:</strong> This is a research and educational demonstration only. 
    This system is NOT intended for actual fraud detection in production environments. 
    Results may be inaccurate and should not be used for investment or business decisions.
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models and feature engineer."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        st.error("Models not found! Please run the training script first: `python scripts/train.py`")
        return None, None
    
    try:
        # Load feature engineer
        feature_engineer_path = models_dir / "feature_engineer.pkl"
        if feature_engineer_path.exists():
            feature_engineer = joblib.load(feature_engineer_path)
        else:
            st.error("Feature engineer not found!")
            return None, None
        
        # Load models
        models = {}
        model_files = {
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'isolation_forest': 'isolation_forest_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                if model_name == 'xgboost':
                    models[model_name] = XGBoostFraudDetector({})
                elif model_name == 'lightgbm':
                    models[model_name] = LightGBMFraudDetector({})
                elif model_name == 'isolation_forest':
                    models[model_name] = IsolationForestFraudDetector({})
                
                models[model_name].load_model(str(model_path))
            else:
                st.warning(f"Model {model_name} not found!")
        
        return models, feature_engineer
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration."""
    config = ClaimsDataConfig(n_samples=1000, fraud_rate=0.15, random_seed=42)
    generator = SyntheticClaimsData(config)
    return generator.generate_data()

def create_feature_input_form():
    """Create input form for manual claim entry."""
    st.subheader("📝 Manual Claim Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        claim_amount = st.number_input(
            "Claim Amount ($)", 
            min_value=100, 
            max_value=100000, 
            value=5000,
            step=100
        )
        
        customer_age = st.slider(
            "Customer Age", 
            min_value=18, 
            max_value=80, 
            value=45
        )
        
        num_previous_claims = st.slider(
            "Number of Previous Claims", 
            min_value=0, 
            max_value=10, 
            value=1
        )
        
    with col2:
        claim_type = st.selectbox(
            "Claim Type",
            ["Accident", "Fire", "Theft", "Health", "Property"]
        )
        
        location = st.selectbox(
            "Location",
            ["Urban", "Suburban", "Rural"]
        )
        
        customer_segment = st.selectbox(
            "Customer Segment",
            ["Premium", "Standard", "Basic"]
        )
        
        claim_time = st.selectbox(
            "Claim Time",
            ["Business_Hours", "Evening", "Night", "Weekend"]
        )
    
    # Additional features
    days_since_last_claim = st.number_input(
        "Days Since Last Claim", 
        min_value=0, 
        max_value=2000, 
        value=365,
        step=1
    )
    
    policy_duration = st.number_input(
        "Policy Duration (years)", 
        min_value=0.1, 
        max_value=20.0, 
        value=5.0,
        step=0.1
    )
    
    return {
        'claim_amount': claim_amount,
        'customer_age': customer_age,
        'num_previous_claims': num_previous_claims,
        'days_since_last_claim': days_since_last_claim,
        'policy_duration': policy_duration,
        'claim_type': claim_type,
        'location': location,
        'customer_segment': customer_segment,
        'claim_time': claim_time
    }

def predict_fraud(models, feature_engineer, claim_data):
    """Predict fraud for a given claim."""
    # Create DataFrame
    df = pd.DataFrame([claim_data])
    
    # Add derived features
    df['claim_amount_per_previous'] = df['claim_amount'] / (df['num_previous_claims'] + 1)
    df['age_group'] = pd.cut(df['customer_age'], bins=[0, 25, 35, 50, 65, 100], labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
    df['high_value_claim'] = (df['claim_amount'] > df['claim_amount'].quantile(0.9)).astype(int)
    df['frequent_claimant'] = (df['num_previous_claims'] >= 3).astype(int)
    df['recent_claim'] = (df['days_since_last_claim'] < 30).astype(int)
    
    # Process features
    X_processed, _ = feature_engineer.prepare_features(df, fit=False)
    
    # Get predictions from all models
    predictions = {}
    probabilities = {}
    
    for model_name, model in models.items():
        if model.is_fitted:
            pred = model.predict(X_processed)[0]
            proba = model.predict_proba(X_processed)[0]
            predictions[model_name] = pred
            probabilities[model_name] = proba
    
    return predictions, probabilities

def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Load models
    models, feature_engineer = load_models()
    
    if models is None or feature_engineer is None:
        st.stop()
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Manual Claim Entry", "Batch Analysis", "Model Comparison"]
    )
    
    if mode == "Manual Claim Entry":
        st.header("🔍 Single Claim Analysis")
        
        # Get claim data from user
        claim_data = create_feature_input_form()
        
        if st.button("🔍 Analyze Claim", type="primary"):
            # Make predictions
            predictions, probabilities = predict_fraud(models, feature_engineer, claim_data)
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Create columns for results
            cols = st.columns(len(models))
            
            for i, (model_name, model) in enumerate(models.items()):
                if model.is_fitted:
                    with cols[i]:
                        pred = predictions[model_name]
                        proba = probabilities[model_name]
                        
                        st.metric(
                            label=f"{model_name.upper()} Prediction",
                            value="🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE",
                            delta=f"Probability: {proba:.3f}"
                        )
            
            # Overall assessment
            avg_probability = np.mean(list(probabilities.values()))
            overall_prediction = 1 if avg_probability > 0.5 else 0
            
            st.subheader("🎯 Overall Assessment")
            
            if overall_prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>🚨 HIGH FRAUD RISK</h3>
                    <p>Average Fraud Probability: <strong>{avg_probability:.3f}</strong></p>
                    <p>Recommendation: <strong>Investigate this claim</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legitimate-claim">
                    <h3>✅ LOW FRAUD RISK</h3>
                    <p>Average Fraud Probability: <strong>{avg_probability:.3f}</strong></p>
                    <p>Recommendation: <strong>Process normally</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance (if available)
            if 'xgboost' in models and models['xgboost'].is_fitted:
                st.subheader("📈 Feature Importance")
                importance = models['xgboost'].get_feature_importance()
                
                # Sort by importance
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Create bar chart
                fig = px.bar(
                    x=[x[1] for x in sorted_importance[:10]],
                    y=[x[0] for x in sorted_importance[:10]],
                    orientation='h',
                    title="Top 10 Most Important Features",
                    labels={'x': 'Importance Score', 'y': 'Feature'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        
        # Generate sample data
        sample_data = generate_sample_data()
        
        st.subheader("Sample Claims Data")
        st.dataframe(sample_data.head(10))
        
        # Analyze sample
        if st.button("🔍 Analyze Sample Claims"):
            # Process features
            X_processed, _ = feature_engineer.prepare_features(sample_data, fit=False)
            
            # Get predictions
            results = {}
            for model_name, model in models.items():
                if model.is_fitted:
                    predictions = model.predict(X_processed)
                    probabilities = model.predict_proba(X_processed)
                    results[model_name] = {
                        'predictions': predictions,
                        'probabilities': probabilities
                    }
            
            # Display results
            st.subheader("📈 Batch Analysis Results")
            
            # Create summary
            summary_data = []
            for model_name, result in results.items():
                fraud_count = result['predictions'].sum()
                avg_prob = result['probabilities'].mean()
                summary_data.append({
                    'Model': model_name.upper(),
                    'Fraud Detected': fraud_count,
                    'Avg Fraud Probability': f"{avg_prob:.3f}",
                    'Detection Rate': f"{fraud_count/len(sample_data)*100:.1f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Fraud Distribution', 'Probability Distribution', 'Model Comparison', 'Feature Analysis'],
                specs=[[{"type": "pie"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Fraud distribution
            fraud_counts = [result['predictions'].sum() for result in results.values()]
            fig.add_trace(
                go.Pie(labels=list(results.keys()), values=fraud_counts, name="Fraud Distribution"),
                row=1, col=1
            )
            
            # Probability distribution
            all_probs = np.concatenate([result['probabilities'] for result in results.values()])
            fig.add_trace(
                go.Histogram(x=all_probs, name="Probability Distribution"),
                row=1, col=2
            )
            
            # Model comparison
            model_names = list(results.keys())
            avg_probs = [result['probabilities'].mean() for result in results.values()]
            fig.add_trace(
                go.Bar(x=model_names, y=avg_probs, name="Average Probability"),
                row=2, col=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Model Comparison":
        st.header("⚖️ Model Comparison")
        
        st.subheader("Model Performance Metrics")
        
        # Generate test data
        test_data = generate_sample_data()
        X_test = test_data.drop('fraudulent_claim', axis=1)
        y_test = test_data['fraudulent_claim']
        
        # Process features
        X_test_processed, _ = feature_engineer.prepare_features(X_test, fit=False)
        
        # Evaluate models
        evaluator = FraudDetectionEvaluator({
            'metrics': ['auc', 'precision', 'recall', 'f1', 'accuracy'],
            'k_values': [10, 50, 100]
        })
        
        comparison_data = []
        
        for model_name, model in models.items():
            if model.is_fitted:
                y_pred = model.predict(X_test_processed)
                y_proba = model.predict_proba(X_test_processed)
                
                metrics = evaluator.evaluate(y_test, y_pred, y_proba)
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'AUC': f"{metrics.get('auc', 0):.4f}",
                    'Precision': f"{metrics.get('precision', 0):.4f}",
                    'Recall': f"{metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{metrics.get('f1', 0):.4f}",
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance visualization
        metrics_to_plot = ['AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric].astype(float),
                text=comparison_df[metric],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>🛡️ Claims Fraud Detection System - Research & Educational Use Only</p>
        <p>⚠️ This system is not intended for production use or investment decisions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
