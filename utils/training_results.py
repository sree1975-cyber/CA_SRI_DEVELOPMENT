"""
Training Results tab functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.common import save_model, plot_feature_importance

def render_training_results_tab():
    """Render the Training Results tab content"""
    st.markdown("<div class='card-title'>📈 Training Results & Model Performance</div>", unsafe_allow_html=True)
    
    # Display training results if available
    if st.session_state.model is not None and st.session_state.training_report is not None:
        # Model information
        st.markdown("### Model Information")
        
        # Determine model type and display appropriate information
        model_type = st.session_state.active_model if 'active_model' in st.session_state else 'unknown'
        model_name = model_type.replace('_', ' ').title() if model_type else 'Unknown'
        
        # Show model summary
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.metric("Model Type", model_name)
            
            if hasattr(st.session_state.model, 'n_estimators'):
                st.metric("Number of Trees", st.session_state.model.n_estimators)
            
            if hasattr(st.session_state.model, 'n_features_in_'):
                st.metric("Features Used", st.session_state.model.n_features_in_)
        
        with model_col2:
            if 'training_report' in st.session_state and st.session_state.training_report:
                st.metric("Accuracy", f"{st.session_state.training_report['accuracy']:.4f}")
                st.metric("F1 Score", f"{st.session_state.training_report['f1_score']:.4f}")
        
        # Performance metrics section
        st.markdown("### Model Performance")
        
        # Create a bar chart for metrics
        metrics = {
            'Accuracy': st.session_state.training_report['accuracy'],
            'Precision': st.session_state.training_report['precision'],
            'Recall': st.session_state.training_report['recall'],
            'F1 Score': st.session_state.training_report['f1_score']
        }
        
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            color='Value',
            color_continuous_scale='Viridis',
            range_y=[0, 1],
            labels={'Value': 'Score'},
            title='Model Performance Metrics'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix if available
        if 'confusion_matrix' in st.session_state.training_report:
            st.markdown("### Confusion Matrix")
            
            cm = st.session_state.training_report['confusion_matrix']
            
            # Labels for the confusion matrix
            labels = ['Not CA', 'CA']
            
            # Create a heatmap for the confusion matrix
            cm_fig = px.imshow(
                cm,
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale='Blues'
            )
            
            cm_fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Explain the confusion matrix
            st.markdown("""
            **Understanding the Confusion Matrix:**
            - **Top-Left (True Negative)**: Correctly predicted non-chronic absenteeism
            - **Top-Right (False Positive)**: Incorrectly predicted chronic absenteeism
            - **Bottom-Left (False Negative)**: Missed actual chronic absenteeism
            - **Bottom-Right (True Positive)**: Correctly predicted chronic absenteeism
            """)
        
        # Feature importance visualization
        st.markdown("### Feature Importance")
        
        # Create feature importance plot
        fi_fig = plot_feature_importance(st.session_state.model)
        
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)
            
            # Explain feature importance
            st.markdown("""
            **Feature importance** shows which factors have the strongest influence on predicting chronic absenteeism.
            Higher values indicate greater importance in the model's decision-making process.
            """)
        else:
            st.info("Feature importance visualization is not available for this model type.")
        
        # Model export option
        st.markdown("### Export Model")
        st.markdown("""
        You can download the trained model for use in other systems or for future reference.
        The model is saved in joblib format, which preserves all the trained parameters.
        """)
        
        # Save model button in a container with styling
        save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
        with save_col2:
            save_model()
    else:
        st.info("⏳ No training results available yet. Please train a model first.")
        
        # Show a placeholder visualization
        st.markdown("### Ready to Train")
        st.markdown("""
        Once you train a model, you'll see:
        - Performance metrics (accuracy, precision, recall, F1 score)
        - Feature importance visualization
        - Option to export the trained model
        """)
    
    return