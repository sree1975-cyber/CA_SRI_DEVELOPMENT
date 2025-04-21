"""
Model Parameters tab functionality
"""

import streamlit as st
from utils.common import train_models

def render_model_params_tab():
    """Render the Model Parameters tab content"""
    st.markdown("<div class='card-title'>⚙️ Model Parameters</div>", unsafe_allow_html=True)
    
    # Model selection and parameters form
    with st.form(key="model_params_form"):
        st.markdown("### Select Model Type")
        
        # Model selection
        model_type = st.selectbox(
            "Machine Learning Model",
            options=[
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
                "Neural Network"
            ],
            index=0
        )
        
        # Convert display name to internal name
        model_type_key = model_type.lower().replace(" ", "_")
        
        # Store the selected model type
        if 'active_model' not in st.session_state or st.session_state.active_model != model_type_key:
            st.session_state.active_model = model_type_key
        
        # Model parameters - we'll show different parameters based on the model type
        st.markdown("### Configure Model Parameters")
        
        # Default parameters based on model type
        if model_type_key == "random_forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
            max_depth = st.slider("Maximum Tree Depth", 2, 30, 10, 1)
            min_samples_split = st.slider("Minimum Samples to Split", 2, 20, 2, 1)
            
            model_params = {
                'random_forest': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'random_state': 42
                }
            }
        
        elif model_type_key == "gradient_boosting":
            n_estimators = st.slider("Number of Boosting Stages", 10, 500, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Maximum Tree Depth", 2, 10, 3, 1)
            
            model_params = {
                'gradient_boost': {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'random_state': 42
                }
            }
        
        elif model_type_key == "logistic_regression":
            c_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
            max_iter = st.slider("Maximum Iterations", 100, 2000, 1000, 100)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag"], index=0)
            
            model_params = {
                'logistic_regression': {
                    'C': c_value,
                    'max_iter': max_iter,
                    'solver': solver,
                    'random_state': 42
                }
            }
        
        elif model_type_key == "neural_network":
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes", "50, 25")
            activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"], index=0)
            alpha = st.slider("L2 Regularization", 0.0001, 0.01, 0.0001, 0.0001)
            max_iter = st.slider("Maximum Iterations", 100, 2000, 1000, 100)
            
            # Parse the hidden layer sizes
            try:
                hidden_layers = tuple(int(x.strip()) for x in hidden_layer_sizes.split(","))
            except:
                hidden_layers = (50, 25)
            
            model_params = {
                'neural_network': {
                    'hidden_layer_sizes': hidden_layers,
                    'activation': activation,
                    'alpha': alpha,
                    'max_iter': max_iter,
                    'random_state': 42
                }
            }
        
        # Training parameters
        st.markdown("### Training Parameters")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
        
        # Train button
        train_button = st.form_submit_button(label="Train Model")
    
    # Handle training process if button is clicked
    if train_button:
        # Check if we have training data
        if 'historical_data' not in st.session_state or st.session_state.historical_data.empty:
            st.error("❌ No training data available. Please upload data in the Training Data tab.")
            return
        
        # Set training status
        st.session_state.training_status = "in_progress"
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Status message
        status_text = st.empty()
        status_text.text("Starting training process...")
        
        # Update progress
        import time
        for i in range(101):
            if i < 20:
                status_text.text("Preprocessing data...")
            elif i < 40:
                status_text.text("Splitting training and test sets...")
            elif i < 70:
                status_text.text(f"Training {model_type} model...")
            elif i < 90:
                status_text.text("Evaluating model performance...")
            else:
                status_text.text("Finalizing model...")
            
            progress_bar.progress(i)
            time.sleep(0.05)
        
        # Train the model
        status_text.text(f"Training {model_type} model with {len(st.session_state.historical_data)} records...")
        
        # Call train_models with the selected model type and parameters
        training_results = train_models(
            st.session_state.historical_data, 
            models_to_train=[model_type_key],
            params=model_params
        )
        
        if training_results and 'models' in training_results and model_type_key in training_results['models']:
            # Store the model and related information in the session state
            st.session_state.model = training_results['models'][model_type_key]
            st.session_state.feature_names = training_results['feature_names']
            st.session_state.training_report = training_results['reports'][model_type_key]
            
            # Set training status to complete
            st.session_state.training_status = "complete"
            
            # Success message
            st.success(f"✅ {model_type} model trained successfully!")
            
            # Show key metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Accuracy", f"{training_results['reports'][model_type_key]['accuracy']:.4f}")
            with metrics_col2:
                st.metric("Precision", f"{training_results['reports'][model_type_key]['precision']:.4f}")
            with metrics_col3:
                st.metric("Recall", f"{training_results['reports'][model_type_key]['recall']:.4f}")
            with metrics_col4:
                st.metric("F1 Score", f"{training_results['reports'][model_type_key]['f1_score']:.4f}")
            
        else:
            # Set training status to failed
            st.session_state.training_status = "failed"
            
            # Error message
            st.error("❌ Model training failed. Please check your data and parameters.")
    
    # Show advanced options in an expander
    with st.expander("Advanced Options"):
        st.markdown("### Feature Selection")
        st.markdown("""
        By default, all available features are used for training. In a future version, 
        you'll be able to select specific features to include in your model.
        """)
        
        st.markdown("### Hyperparameter Tuning")
        st.markdown("""
        For optimal performance, consider using hyperparameter tuning. In a future version, 
        you'll be able to automatically search for the best parameters for your model.
        """)
    
    # Show model information in an expander
    with st.expander("Model Information"):
        st.markdown(f"### {model_type}")
        
        if model_type_key == "random_forest":
            st.markdown("""
            **Random Forest** is an ensemble learning method that constructs multiple decision trees 
            during training and outputs the class that is the mode of the classes of the individual trees.
            
            *Advantages:*
            - Performs well on many problems, including complex datasets
            - Robust to outliers and non-linear data
            - Provides feature importance metrics
            
            *Parameters:*
            - **Number of Trees**: More trees generally result in better performance but slower training
            - **Maximum Tree Depth**: Controls the maximum depth of each tree; deeper trees may overfit
            - **Minimum Samples to Split**: The minimum number of samples required to split a node
            """)
        
        elif model_type_key == "gradient_boosting":
            st.markdown("""
            **Gradient Boosting** builds an ensemble of decision trees sequentially, with each new tree 
            correcting errors made by previously trained trees.
            
            *Advantages:*
            - Often provides higher accuracy than single models
            - Handles mixed data types well
            - Robust to outliers and missing data
            
            *Parameters:*
            - **Number of Boosting Stages**: The number of sequential trees to build
            - **Learning Rate**: Controls how much each tree contributes to the final result
            - **Maximum Tree Depth**: Controls complexity of the constituent trees
            """)
        
        elif model_type_key == "logistic_regression":
            st.markdown("""
            **Logistic Regression** is a linear model that predicts the probability of an observation 
            belonging to a certain class. It's simple, interpretable, and works well for linearly separable data.
            
            *Advantages:*
            - Simple and interpretable
            - Works well for linearly separable data
            - Less prone to overfitting than complex models
            
            *Parameters:*
            - **Regularization Strength (C)**: Inverse of regularization strength; smaller values increase regularization
            - **Maximum Iterations**: Maximum number of iterations for the solver to converge
            - **Solver**: Algorithm to use in the optimization problem
            """)
        
        elif model_type_key == "neural_network":
            st.markdown("""
            **Neural Network** is a deep learning approach that can model complex relationships between inputs and outputs.
            We use a Multi-Layer Perceptron (MLP) classifier for this application.
            
            *Advantages:*
            - Can capture complex non-linear patterns
            - Handles high-dimensional data well
            - Adaptable to many types of problems
            
            *Parameters:*
            - **Hidden Layer Sizes**: The number of neurons in each hidden layer
            - **Activation Function**: The function used to transform the output of each neuron
            - **L2 Regularization**: Penalizes large weights to prevent overfitting
            - **Maximum Iterations**: Maximum number of iterations for the solver to converge
            """)
    
    return