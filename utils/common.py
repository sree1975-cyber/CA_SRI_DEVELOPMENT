"""
Common utilities shared across different modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
import base64
from datetime import datetime

def display_svg(file_path, width=None):
    """Display an SVG file in a Streamlit app"""
    if not os.path.exists(file_path):
        # Return a default placeholder if the file doesn't exist
        return f"<div style='text-align: center; color: #888;'>Image not found: {file_path}</div>"
        
    with open(file_path, "r") as f:
        content = f.read()
        
    if width:
        # Add width attribute to the SVG tag
        content = content.replace("<svg ", f"<svg width='{width}' ")
        
    return content

import numpy as np
import pandas as pd

def generate_sample_data():
    """Generate sample data with specific attendance patterns"""
    np.random.seed(42)  # For reproducibility
    
    # Define parameters
    num_students = 200
    current_year = 2023
    total_school_days = 180
    
    # Student IDs
    student_ids = [f"STU{i:04d}" for i in range(1, num_students + 1)]
    
    # Schools
    schools = ["North High", "South High", "East Middle", "West Elementary", "Central Academy"]
    school_data = np.random.choice(schools, num_students)
    
    # Grades (6-12)
    grades = np.random.randint(6, 13, num_students)
    
    # Gender
    genders = np.random.choice(["Male", "Female"], num_students)
    
    # Create specific attendance patterns
    attendance_groups = [
        (0.20, 0.50),   # Below 50% attendance
        (0.75, 0.80),   # 75-80% attendance
        (0.89, 0.89),   # Exactly 89% attendance
        (0.91, 1.00)    # Above 90% attendance
    ]
    
    # Assign students to attendance groups
    group_assignments = np.random.choice(len(attendance_groups), num_students, p=[0.1, 0.3, 0.3, 0.3])
    
    present_days = []
    absent_days = []
    attendance_pct = []
    
    for group_idx in group_assignments:
        low, high = attendance_groups[group_idx]
        pct = np.random.uniform(low, high)
        present = int(total_school_days * pct)
        absent = total_school_days - present
        
        present_days.append(present)
        absent_days.append(absent)
        attendance_pct.append(pct * 100)
    
    # Meal codes - random assignment
    meal_codes = np.random.choice(["Free", "Reduced", "Paid"], num_students, p=[0.4, 0.2, 0.4])
    
    # Academic performance - random with slight attendance correlation
    academic_perf = np.clip(
        np.random.normal(75, 15, num_students) + 
        (np.array(attendance_pct) - 85, 0, 100
    ).astype(int)
    
    # Create the dataframe
    data = pd.DataFrame({
        'Student_ID': student_ids,
        'School': school_data,
        'Grade': grades,
        'Gender': genders,
        'Present_Days': present_days,
        'Absent_Days': absent_days,
        'Attendance_Percentage': attendance_pct,
        'Meal_Code': meal_codes,
        'Academic_Performance': academic_perf,
        'Year': current_year,
        'CA_Status': (np.array(attendance_pct) <= 90).astype(int)  # 1 if <=90%, else 0
    })
    
    # Generate 3 years of historical data maintaining similar attendance patterns
    historical_years = 3
    historical_data = []
    
    for year in range(current_year - historical_years, current_year):
        # Select 60% of students for historical data
        selected_students = np.random.choice(num_students, int(num_students * 0.6), replace=False)
        
        for idx in selected_students:
            student_row = data.iloc[idx].copy()
            year_diff = current_year - year
            student_row['Grade'] = max(6, student_row['Grade'] - year_diff)
            
            if student_row['Grade'] >= 6:
                # Slightly modify attendance while keeping in same general range
                current_pct = student_row['Attendance_Percentage'] / 100
                
                if current_pct <= 0.5:
                    new_pct = np.random.uniform(0.2, 0.5)
                elif 0.75 <= current_pct <= 0.80:
                    new_pct = np.random.uniform(0.75, 0.80)
                elif current_pct == 0.89:
                    new_pct = np.random.uniform(0.88, 0.90)
                else:  # > 0.90
                    new_pct = np.random.uniform(0.91, 1.0)
                
                new_present = int(total_school_days * new_pct)
                new_absent = total_school_days - new_present
                
                student_row['Present_Days'] = new_present
                student_row['Absent_Days'] = new_absent
                student_row['Attendance_Percentage'] = new_pct * 100
                
                # Slightly modify academic performance
                student_row['Academic_Performance'] = np.clip(
                    student_row['Academic_Performance'] + np.random.normal(0, 5), 0, 100
                ).astype(int)
                
                student_row['Year'] = year
                student_row['CA_Status'] = 1 if new_pct <= 0.90 else 0
                
                historical_data.append(student_row)
    
    # Combine current and historical data
    if historical_data:
        historical_df = pd.DataFrame(historical_data)
        combined_data = pd.concat([data, historical_df])
    else:
        combined_data = data
    
    return data, combined_data

def preprocess_data(df, is_training=True):
    """Preprocess the input data for training or prediction"""
    try:
        processed_df = df.copy()
        
        # Handle missing values
        numerical_cols = ['Present_Days', 'Absent_Days', 'Attendance_Percentage', 'Academic_Performance']
        for col in numerical_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # One-hot encode categorical variables
        categorical_cols = ['School', 'Gender', 'Meal_Code']
        categorical_cols = [col for col in categorical_cols if col in processed_df.columns]
        
        for col in categorical_cols:
            dummies = pd.get_dummies(processed_df[col], prefix=col, dummy_na=True)
            processed_df = pd.concat([processed_df, dummies], axis=1)
            processed_df.drop(col, axis=1, inplace=True)
        
        if is_training:
            drop_cols = ['Student_ID', 'Year']
            drop_cols = [col for col in drop_cols if col in processed_df.columns]
            processed_df.drop(drop_cols, axis=1, inplace=True)
            
            if 'CA_Status' in processed_df.columns:
                if processed_df['CA_Status'].dtype == 'object':
                    processed_df['CA_Status'] = processed_df['CA_Status'].apply(
                        lambda x: 1 if x == 'CA' or x == 1 else 0)
            
        return processed_df
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return None

def train_models(df, models_to_train=['random_forest'], params=None):
    """Train multiple models on the provided data
    
    Args:
        df: DataFrame with the training data
        models_to_train: List of model types to train ('random_forest', 'gradient_boost', 'logistic_regression', 'neural_network')
        params: Dictionary with model parameters (optional)
    
    Returns:
        Dictionary of trained models, feature names, and performance reports
    """
    try:
        # Preprocess the data
        processed_df = preprocess_data(df)
        if processed_df is None:
            return None
        
        # Split features and target - handle both column name options
        target_column = 'CA_Status' if 'CA_Status' in processed_df.columns else 'CA_Label'
        X = processed_df.drop(target_column, axis=1) 
        y = processed_df[target_column]
        
        # Feature names for later use
        feature_names = X.columns.tolist()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Scale only numerical columns
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # Initialize results
        results = {
            'models': {},
            'feature_names': feature_names,
            'preprocessing': {
                'scaler': scaler,
                'numerical_cols': numerical_cols.tolist()
            },
            'reports': {}
        }
        
        # Train selected models
        for model_type in models_to_train:
            # Initialize the model with default or provided parameters
            if model_type == 'random_forest':
                model_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
                if params and 'random_forest' in params:
                    model_params.update(params['random_forest'])
                model = RandomForestClassifier(**model_params)
            
            elif model_type == 'gradient_boost':
                model_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
                if params and 'gradient_boost' in params:
                    model_params.update(params['gradient_boost'])
                model = GradientBoostingClassifier(**model_params)
            
            elif model_type == 'logistic_regression':
                model_params = {'max_iter': 1000, 'random_state': 42}
                if params and 'logistic_regression' in params:
                    model_params.update(params['logistic_regression'])
                model = LogisticRegression(**model_params)
            
            elif model_type == 'neural_network':
                model_params = {'hidden_layer_sizes': (50, 25), 'max_iter': 1000, 'random_state': 42}
                if params and 'neural_network' in params:
                    model_params.update(params['neural_network'])
                model = MLPClassifier(**model_params)
            
            else:
                st.warning(f"Unknown model type: {model_type}")
                continue
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = None
            
            # Get class probabilities if the model supports it
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            report = classification_report(y_test, y_pred, zero_division=0)
            
            # Save model and performance metrics
            results['models'][model_type] = model
            results['reports'][model_type] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': report,
                'y_pred_proba': y_pred_proba
            }
            
            # Calculate feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['reports'][model_type]['feature_importance'] = dict(
                    zip(feature_names, model.feature_importances_)
                )
            elif hasattr(model, 'coef_') and model.coef_.shape[0] == 1:
                results['reports'][model_type]['feature_importance'] = dict(
                    zip(feature_names, model.coef_[0])
                )
        
        return results
    
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

def train_model(df):
    """Legacy function for backward compatibility"""
    try:
        results = train_models(df, models_to_train=['random_forest'])
        if results and 'models' in results and 'random_forest' in results['models']:
            model = results['models']['random_forest']
            # Store the target column name we used in the model as an attribute
            target_column = 'CA_Status' if 'CA_Status' in df.columns else 'CA_Label'
            model.target_column = target_column
            return model, results['feature_names'], results['reports']['random_forest']
        return None, None, None
    except Exception as e:
        st.error(f"Error in train_model: {str(e)}")
        return None, None, None

def predict_ca_risk(input_data, model):
    """Predict CA risk for input data with proper error handling"""
    try:
        # Ensure the input is a DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Check if model is available
        if model is None:
            st.error("No trained model available. Please train a model first.")
            return None
        
        # Preprocess the input data
        processed_input = preprocess_data(input_df, is_training=False)
        if processed_input is None:
            return None
        
        # Feature selection for prediction
        feature_cols = []
        
        # For traditional sklearn models
        if hasattr(model, 'feature_names_in_'):
            feature_cols = list(model.feature_names_in_)  # Convert to list to be safe
        # For ensemble models like RandomForest
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            if hasattr(model.estimators_[0], 'feature_names_in_'):
                feature_cols = list(model.estimators_[0].feature_names_in_)
        
        # If feature names can't be determined from the model
        if not feature_cols:
            # Get features from the first estimator in ensemble models
            if hasattr(model, 'n_features_in_'):
                # Try to infer matching columns by position (risky)
                feature_cols = list(processed_input.columns[:model.n_features_in_])
            else:
                # Last resort: exclude known non-feature columns
                exclude_cols = ['Student_ID', 'Year', 'CA_Status', 'CA_Label', 'CA_Risk']
                feature_cols = [col for col in processed_input.columns if col not in exclude_cols]
        
        # Select only the features needed for prediction
        X_pred = processed_input.copy()
        
        # Make sure all needed columns are present (with 0s for missing ones)
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        try:
            # This line sometimes triggers the "truth value of array" error - wrap in try/except
            # Explicitly convert feature_cols to a list to avoid array comparison issues
            feature_cols_list = list(feature_cols)
            # Only select columns that actually exist in the DataFrame
            valid_cols = [col for col in feature_cols_list if col in X_pred.columns]
            X_pred = X_pred[valid_cols]
        
        except Exception as e:
            # If there's any issue with column selection, let's try an alternative approach
            st.warning(f"Column selection issue detected: {str(e)}")
            # Exclude non-feature columns instead
            exclude_cols = ['Student_ID', 'Year', 'CA_Status', 'CA_Label', 'CA_Risk']
            X_pred = X_pred.drop([col for col in exclude_cols if col in X_pred.columns], axis=1)
        
        # Scale numerical features if scaler is available
        try:
            if hasattr(model, 'preprocessing_') and 'scaler' in model.preprocessing_:
                scaler = model.preprocessing_['scaler']
                numerical_cols = model.preprocessing_['numerical_cols']
                numerical_cols_present = [col for col in numerical_cols if col in X_pred.columns]
                if numerical_cols_present:
                    X_pred[numerical_cols_present] = scaler.transform(X_pred[numerical_cols_present])
        except Exception as e:
            st.warning(f"Scaling issue: {str(e)}")
            # Continue without scaling if there's an error
        
        # Make prediction
        try:
            if hasattr(model, 'predict_proba'):
                # Get probability of the positive class
                risk_probabilities = model.predict_proba(X_pred)[:, 1]
                
                # Convert numpy arrays to Python native types to avoid comparison issues
                if hasattr(risk_probabilities, 'shape'):
                    if risk_probabilities.shape[0] == 1:
                        return float(risk_probabilities[0])
                    else:
                        return risk_probabilities.tolist()  # Convert to list for multi-predictions
                
                # If not a numpy array, return as is
                return risk_probabilities
                
            else:
                # If the model doesn't support probabilities, return the binary prediction
                risk_prediction = model.predict(X_pred)
                
                # Convert to native Python types to avoid comparison issues
                if hasattr(risk_prediction, 'shape'):
                    if risk_prediction.shape[0] == 1:
                        return float(risk_prediction[0])
                    else:
                        return risk_prediction.tolist()  # Convert to list for multi-predictions
                
                # If not a numpy array, return as is
                return risk_prediction
        
        except Exception as e:
            st.error(f"Error in model prediction: {str(e)}")
            return None
    
    except Exception as e:
        st.error(f"Error in prediction pre-processing: {str(e)}")
        return None

def plot_risk_gauge(risk_value, key=None):
    """Create a gauge chart for risk visualization"""
    if risk_value is None:
        return None
    
    # Convert to scalar if it's an array (handles both single and batch predictions)
    if hasattr(risk_value, '__len__') and len(risk_value) > 0 and not isinstance(risk_value, (str, dict)):
        risk_value = float(risk_value[0])
    
    # Make sure risk_value is a valid float
    try:
        risk_value_float = float(risk_value)
    except (ValueError, TypeError):
        st.error(f"Invalid risk value: {risk_value} cannot be converted to float")
        return None
        
    # Determine gauge color based on risk level
    if risk_value_float < 0.3:
        color = "green"
    elif risk_value_float < 0.7:
        color = "gold"
    else:
        color = "red"
    
    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_value_float * 100,  # Convert to percentage
        title = {'text': "CA Risk Level"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(255, 215, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    
    return fig

def plot_feature_importance(model, key=None):
    """Create interactive feature importance visualization"""
    if not model or not hasattr(model, 'feature_importances_'):
        return None
        
    # Get feature importances and names
    importances = model.feature_importances_
    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"Feature {i}" for i in range(len(importances))]
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top features
    top_features = importance_df.head(10)
    
    # Create the bar chart
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=10),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_student_history(student_id):
    """Plot historical trends for a student"""
    if 'historical_data' not in st.session_state or st.session_state.historical_data.empty:
        return None
        
    # Filter data for the specific student
    student_data = st.session_state.historical_data[st.session_state.historical_data['Student_ID'] == student_id].copy()
    
    if student_data.empty:
        return None
    
    # Sort by Year
    student_data = student_data.sort_values('Year')
    
    # Create risk trend
    risk_fig = px.line(
        student_data,
        x='Year',
        y='CA_Risk',
        title='Risk Level Trend',
        markers=True,
        line_shape='linear'
    )
    
    risk_fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Risk Score',
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    
    # Create attendance trend
    attendance_fig = px.line(
        student_data,
        x='Year',
        y='Attendance_Percentage',
        title='Attendance Trend',
        markers=True,
        line_shape='linear'
    )
    
    attendance_fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Attendance (%)',
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    
    # Combine the figures
    combined_fig = go.Figure()
    
    # Add risk trace
    for trace in risk_fig.data:
        trace.name = "Risk Score"
        trace.yaxis = "y"
        combined_fig.add_trace(trace)
    
    # Add attendance trace    
    for trace in attendance_fig.data:
        trace.name = "Attendance (%)"
        trace.yaxis = "y2"
        combined_fig.add_trace(trace)
    
    # Update layout to include both y-axes
    combined_fig.update_layout(
        title='Student History',
        xaxis_title='Year',
        yaxis=dict(
            title="Risk Score",
            range=[0, 1],
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title="Attendance (%)",
            range=[0, 100],
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=30)
    )
    
    return combined_fig

def get_recommendation(risk_value, what_if=False):
    """Generate recommendations based on risk level"""
    prefix = "What-If " if what_if else ""
    
    # Convert to scalar if it's an array
    if hasattr(risk_value, '__len__') and len(risk_value) > 0 and not isinstance(risk_value, (str, dict)):
        risk_value = float(risk_value[0])
    
    # Make sure risk_value is a valid float
    try:
        risk_value_float = float(risk_value)
    except (ValueError, TypeError):
        st.error(f"Invalid risk value for recommendations: {risk_value} cannot be converted to float")
        return ["Error processing risk level. Please check input data."]
    
    if risk_value_float < 0.3:
        return [
            f"{prefix}Low Risk: No immediate intervention required.",
            "Continue standard attendance monitoring.",
            "Recognize and reinforce positive attendance patterns.",
            "Maintain communication with student and family."
        ]
    elif risk_value_float < 0.7:
        return [
            f"{prefix}Medium Risk: Preventative intervention recommended.",
            "Schedule a check-in meeting with the student.",
            "Contact family to discuss attendance patterns.",
            "Create an attendance plan with clear goals.",
            "Consider academic or social support services."
        ]
    else:
        return [
            f"{prefix}High Risk: Immediate intervention required.",
            "Schedule a comprehensive assessment meeting.",
            "Develop an intensive intervention plan.",
            "Connect with family support services.",
            "Consider home visits or daily check-ins.",
            "Implement academic recovery strategies.",
            "Schedule weekly progress monitoring."
        ]

def on_student_id_change():
    """Handle changes to the student ID field"""
    # Reset the prediction
    if 'current_prediction' in st.session_state:
        st.session_state.current_prediction = None
    if 'calculation_complete' in st.session_state:
        st.session_state.calculation_complete = False
    
    # If we have historical data, attempt to load the student's data into the form
    if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
        student_id = st.session_state.student_id_input
        
        # Safety check - ensure we have a string student ID
        if student_id is None or student_id == "":
            # No valid student ID, just return
            return
                
        # Filter for the student's latest record - with safety check
        if 'Student_ID' in st.session_state.historical_data.columns:
            student_data = st.session_state.historical_data[
                st.session_state.historical_data['Student_ID'] == student_id
            ]
        else:
            # No Student_ID column, can't filter
            print("Warning: No Student_ID column in historical data")
            return
        
        if not student_data.empty:
            # Sort by year and get most recent record
            student_data = student_data.sort_values('Year', ascending=False).iloc[0]
            
            # Update form fields with student data if fields exist in the data
            if 'School' in student_data:
                st.session_state.school_input = student_data['School']
            if 'Grade' in student_data:
                st.session_state.grade_input = int(student_data['Grade'])
            if 'Gender' in student_data:
                st.session_state.gender_input = student_data['Gender']
            if 'Meal_Code' in student_data:
                st.session_state.meal_code_input = student_data['Meal_Code']
            if 'Present_Days' in student_data:
                st.session_state.present_days_input = float(student_data['Present_Days'])
            if 'Absent_Days' in student_data:
                st.session_state.absent_days_input = float(student_data['Absent_Days'])
            if 'Academic_Performance' in student_data:
                st.session_state.academic_perf_input = float(student_data['Academic_Performance'])
                
            # Log success message
            print(f"Loaded data for student {student_id}")
        else:
            # Log that we found no data
            print(f"No historical data found for student {student_id}")

def on_calculate_risk():
    """Calculate risk based on current input fields"""
    if not st.session_state.model:
        st.warning("⚠️ No model available. Please train a model first.")
        return
    
    # Gather input data from session state
    current_inputs = {
        'Student_ID': st.session_state.student_id_input if 'student_id_input' in st.session_state else 'NEW_STUDENT',
        'School': st.session_state.school_input,
        'Grade': st.session_state.grade_input,
        'Gender': st.session_state.gender_input,
        'Present_Days': st.session_state.present_days_input,
        'Absent_Days': st.session_state.absent_days_input,
        'Meal_Code': st.session_state.meal_code_input,
        'Academic_Performance': st.session_state.academic_perf_input
    }
    
    # Add derived fields
    total_days = current_inputs['Present_Days'] + current_inputs['Absent_Days']
    if total_days > 0:
        current_inputs['Attendance_Percentage'] = (current_inputs['Present_Days'] / total_days) * 100
    else:
        current_inputs['Attendance_Percentage'] = 0
    
    # Run prediction
    risk = predict_ca_risk(current_inputs, st.session_state.model)
    if risk is not None:
        # Store the single risk value (not an array)
        st.session_state.current_prediction = risk
        st.session_state.calculation_complete = True
        
        # Store original prediction for what-if comparisons
        st.session_state.original_prediction = risk
        st.session_state.input_data = current_inputs
    else:
        st.error("Error in prediction. Please check inputs and model.")

def on_calculate_what_if():
    """Calculate risk for what-if scenario"""
    if not st.session_state.model:
        st.warning("⚠️ No model available. Please train a model first.")
        return
    
    # Only run if we have original data
    if 'input_data' not in st.session_state:
        st.warning("Original prediction not found. Please calculate a base prediction first.")
        return
    
    # Start with the original inputs
    what_if_inputs = st.session_state.input_data.copy()
    
    # Update with what-if values
    what_if_inputs['Present_Days'] = st.session_state.what_if_present_days
    what_if_inputs['Absent_Days'] = st.session_state.what_if_absent_days
    what_if_inputs['Academic_Performance'] = st.session_state.what_if_academic_perf
    
    # Recalculate derived fields
    total_days = what_if_inputs['Present_Days'] + what_if_inputs['Absent_Days']
    if total_days > 0:
        what_if_inputs['Attendance_Percentage'] = (what_if_inputs['Present_Days'] / total_days) * 100
    else:
        what_if_inputs['Attendance_Percentage'] = 0
    
    # Run prediction
    risk = predict_ca_risk(what_if_inputs, st.session_state.model)
    if risk is not None:
        # Store the risk value (not as array)
        st.session_state.what_if_prediction = risk
        st.session_state.what_if_complete = True
        
        # Show a success message
        st.success("What-if analysis completed successfully!")
    else:
        st.error("Error in what-if prediction. Please check inputs and model.")

def batch_predict_ca(df, model):
    """Run predictions for multiple students"""
    try:
        # Ensure model is available
        if model is None:
            st.warning("⚠️ No model available. Please train a model first.")
            return None
        
        # Make copy to avoid modifying original
        result_df = df.copy()
        
        # Run prediction
        risk_values = predict_ca_risk(result_df, model)
        
        if risk_values is not None:
            # Add risk predictions to the dataframe - ensure it's in the right format
            # Convert to list if it's a numpy array to avoid ambiguity in truth value comparisons
            if hasattr(risk_values, 'tolist'):
                risk_values_list = risk_values.tolist()
            else:
                risk_values_list = risk_values if isinstance(risk_values, list) else [risk_values]
                
            # Add the risk values to the dataframe
            result_df['CA_Risk'] = risk_values_list
            
            # Add risk category based on risk level
            risk_categories = ["Low", "Medium", "High"]
            risk_thresholds = [0.3, 0.7]
            
            # Helper function to categorize each risk value
            def categorize_risk(value):
                try:
                    value_float = float(value)
                    if value_float < 0.3:
                        return "Low"
                    elif value_float < 0.7:
                        return "Medium"
                    else:
                        return "High"
                except (ValueError, TypeError):
                    return "Unknown"
            
            # Apply the categorization function to each risk value
            result_df['Risk_Category'] = result_df['CA_Risk'].apply(categorize_risk)
            
            return result_df
        else:
            return None
    except Exception as e:
        st.error(f"Error in batch prediction: {str(e)}")
        return None

def upload_data_file(file_type="current"):
    """Handle data file uploads
    
    Args:
        file_type: Either "current" for current student data or "historical" for training data
    """
    try:
        uploaded_file = st.file_uploader(
            f"Upload {file_type.title()} Student Data (CSV)", 
            type="csv",
            key=f"{file_type}_data_upload"
        )
        
        if uploaded_file is not None:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Basic validation - Student_ID is now optional
            required_cols = []
            if file_type == "historical":
                required_cols.extend(['School', 'Grade', 'CA_Status', 'Year', 'Present_Days', 'Absent_Days'])
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Process CA_Status for string/numeric format
            if 'CA_Status' in df.columns and df['CA_Status'].dtype == 'object':
                # Convert 'CA'/'NO_CA' to 1/0 if needed
                df['CA_Status'] = df['CA_Status'].apply(lambda x: 1 if x == 'CA' else 0 if x == 'NO_CA' else x)
            
            # Set to session state
            if file_type == "current":
                st.session_state.current_year_data = df
                st.success(f"✅ Successfully loaded {len(df)} student records for current year.")
            else:
                st.session_state.historical_data = df
                st.success(f"✅ Successfully loaded {len(df)} historical student records.")
            
            return df
        
        # Option to use sample data
        if st.button(f"Use Sample {file_type.title()} Data", key=f"sample_{file_type}_data"):
            current_data, historical_data = generate_sample_data()
            
            if file_type == "current":
                st.session_state.current_year_data = current_data
                st.success(f"✅ Successfully loaded {len(current_data)} sample current year records.")
                return current_data
            else:
                st.session_state.historical_data = historical_data
                st.success(f"✅ Successfully loaded {len(historical_data)} sample historical records.")
                return historical_data
        
        return None
    
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def save_model():
    """Save the trained model"""
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("No trained model available to save.")
        return
    
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define filename
        filename = f"ca_predictor_model_{timestamp}.joblib"
        
        # Save the model
        joblib.dump(st.session_state.model, filename)
        
        # Provide download link
        with open(filename, "rb") as f:
            model_bytes = f.read()
        
        b64 = base64.b64encode(model_bytes).decode()
        href = f'<a href="data:file/joblib;base64,{b64}" download="{filename}">Download Trained Model</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.success(f"✅ Model saved successfully as {filename}")
        
        # Clean up the file
        os.remove(filename)
    
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

def generate_system_report():
    """Generate system performance report"""
    if 'training_report' not in st.session_state or st.session_state.training_report is None:
        st.warning("Training report not available. Please train a model first.")
        return False
    
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define filename
        filename = f"ca_predictor_report_{timestamp}.txt"
        
        # Get the report
        report = st.session_state.training_report
        
        # Create report content
        report_content = [
            "CHRONIC ABSENTEEISM PREDICTOR - SYSTEM PERFORMANCE REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n",
            "MODEL INFORMATION:",
            f"Model Type: {st.session_state.active_model.replace('_', ' ').title() if 'active_model' in st.session_state else 'Random Forest'}",
            "\n",
            "PERFORMANCE METRICS:",
            f"Accuracy: {report['accuracy']:.4f}",
            f"Precision: {report['precision']:.4f}",
            f"Recall: {report['recall']:.4f}",
            f"F1 Score: {report['f1_score']:.4f}",
            "\n",
            "CLASSIFICATION REPORT:",
            report['classification_report'],
            "\n",
            "FEATURE IMPORTANCE:",
        ]
        
        # Add feature importance if available
        if 'feature_importance' in report:
            # Sort by importance
            sorted_importance = sorted(
                report['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature, importance in sorted_importance:
                report_content.append(f"{feature}: {importance:.4f}")
        
        # Create 'reports' directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # Save the report to the reports directory
        report_path = os.path.join('reports', filename)
        with open(report_path, "w") as f:
            f.write("\n".join(report_content))
        
        # Save the report path in session state for easy access
        if 'saved_reports' not in st.session_state:
            st.session_state.saved_reports = []
        
        # Add to the list of saved reports (max 10 reports)
        st.session_state.saved_reports.append({
            'filename': filename,
            'path': report_path,
            'content': "\n".join(report_content),
            'timestamp': datetime.now().isoformat(),
            'model_type': st.session_state.active_model if 'active_model' in st.session_state else 'random_forest'
        })
        
        # Keep only the 10 most recent reports
        if len(st.session_state.saved_reports) > 10:
            st.session_state.saved_reports = st.session_state.saved_reports[-10:]
        
        # Provide download link
        with open(report_path, "rb") as f:
            report_bytes = f.read()
        
        b64 = base64.b64encode(report_bytes).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download System Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.success(f"✅ System report generated successfully as {filename}")
        return True
    
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return False
