"""
Chronic Absenteeism Predictor - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import utility modules
from utils.common import (
    display_svg, generate_sample_data, predict_ca_risk,
    plot_risk_gauge, plot_feature_importance, get_recommendation,
    on_student_id_change, on_calculate_risk, on_calculate_what_if
)
from utils.training_data import render_training_data_tab
from utils.model_params import render_model_params_tab
from utils.training_results import render_training_results_tab
from utils.batch_prediction import render_batch_prediction
from utils.advanced_analytics import render_advanced_analytics
from utils.system_settings import render_system_settings

# Set page config
st.set_page_config(
    page_title="Chronic Absenteeism Predictor",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    
    if 'current_year_data' not in st.session_state:
        st.session_state.current_year_data = pd.DataFrame()
    
    if 'training_report' not in st.session_state:
        st.session_state.training_report = None
    
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    
    if 'training_status' not in st.session_state:
        st.session_state.training_status = None
    
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    
    if 'calculation_complete' not in st.session_state:
        st.session_state.calculation_complete = False

# Apply custom CSS
def apply_custom_css():
    """Apply custom CSS styling"""
    css = """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }
    
    .card-subtitle {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background-color: #FFF8E1;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .icon-header {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .icon-header .emoji {
        font-size: 2rem;
    }
    
    .icon-header h2 {
        margin: 0;
    }
    
    .disabled-field {
        opacity: 0.7;
        pointer-events: none;
    }
    
    .recommendation {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Create a sidebar menu
def render_sidebar():
    """Render the sidebar menu"""
    # Logo & title
    if os.path.exists("images/logo.svg"):
        st.sidebar.markdown(display_svg("images/logo.svg", width="100%"), unsafe_allow_html=True)
    
    st.sidebar.title("CA Predictor")
    st.sidebar.markdown("---")
    
    # Navigation
    app_mode = st.sidebar.radio(
        "Navigation",
        options=["System Training", "Batch Prediction", "Advanced Analytics", "System Settings"]
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### System Status")
    
    # Model status - ensure the warning/success indicators match the actual state
    if st.session_state.model is not None:
        model_type = st.session_state.active_model if 'active_model' in st.session_state else 'unknown'
        model_name = model_type.replace('_', ' ').title() if model_type else 'Unknown'
        st.sidebar.success(f"✅ Model: {model_name} (Trained)")
    else:
        st.sidebar.error("❌ Model: Not Trained")
    
    # Data status - match warning/error colors for consistency
    if not st.session_state.historical_data.empty:
        st.sidebar.success(f"✅ Training Data: {len(st.session_state.historical_data)} records")
    else:
        st.sidebar.error("❌ Training Data: Not Loaded")
    
    if not st.session_state.current_year_data.empty:
        st.sidebar.success(f"✅ Current Data: {len(st.session_state.current_year_data)} records")
    else:
        st.sidebar.error("❌ Current Data: Not Loaded")
    
    # System Reset button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("Use this button to completely reset the system and start from scratch.")
    if st.sidebar.button("RESET SYSTEM", help="Clear all data and reset the system"):
        # Reset the session state
        st.session_state.historical_data = pd.DataFrame()
        st.session_state.current_year_data = pd.DataFrame()
        st.session_state.model = None
        if 'training_report' in st.session_state:
            del st.session_state.training_report
        if 'prediction_results' in st.session_state:
            del st.session_state.prediction_results
        
        st.sidebar.success("✅ System reset complete! Please upload new data to begin.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Version 1.0.0")
    st.sidebar.markdown(f"© {datetime.now().year} CA Predictor")
    
    return app_mode

# Individual Student Prediction
def render_individual_prediction():
    """Render the Individual Student Prediction section"""
    # Individual prediction card
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>👨‍🎓 Individual Student Prediction</div>", unsafe_allow_html=True)
    
    # Add an expandable guide for this section
    with st.expander("About Individual Prediction"):
        st.markdown("""
        This section allows you to predict chronic absenteeism risk for a single student.
        
        **How to use:**
        1. Enter the student information in the form
        2. Click 'Calculate CA Risk' to generate a prediction
        3. View the risk level and recommendations
        4. Try 'What-If' scenarios to see how changes might affect the risk level
        
        **Note:** If you enter an existing Student ID that matches historical records, the system will retrieve the student's history for reference.
        """)
    
    # Create a two-column layout for input and result
    col1, col2 = st.columns([2, 1])
    
    with col1:  # Input column
        st.markdown("<div class='card-subtitle'>📝 Student Details</div>", unsafe_allow_html=True)
        
        # Option to select existing student or create new one
        student_sel_options = ["NEW STUDENT"]
        
        # Add existing student IDs if available
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty and 'Student_ID' in st.session_state.historical_data.columns:
            existing_students = st.session_state.historical_data['Student_ID'].unique().tolist()
            student_sel_options.extend(existing_students)
        
        # Safety check - make sure there are valid options (prevents exception)
        if not student_sel_options:
            st.warning("No student data available. Add new students or upload historical data.")
            student_sel_options = ["NEW STUDENT"]
            
        # Student ID selection outside the form to track changes
        try:
            student_select = st.selectbox(
                "Select Student",
                options=student_sel_options,
                index=0,
                key="student_select",
                help="Select an existing student or 'NEW STUDENT' to enter a new one"
            )
        except ValueError as e:
            # Handle case where stored value isn't in the options (e.g. PS-102)
            st.error(f"Student ID selection error: {str(e)}")
            # Reset the selection to avoid the error
            if 'student_select' in st.session_state:
                del st.session_state.student_select
            # Try again with default selection
            student_select = st.selectbox(
                "Select Student",
                options=student_sel_options,
                index=0,
                key="student_select_retry",
                help="Select an existing student or 'NEW STUDENT' to enter a new one"
            )
        
        # If new student selected, show a text input
        if student_select == "NEW STUDENT":
            # Clear any previous student data when switching to NEW STUDENT
            if 'student_id_input' in st.session_state and st.session_state.student_id_input != "":
                # Only if we're changing from an existing student to NEW STUDENT
                if st.session_state.student_id_input != "NEW STUDENT":
                    # Reset form fields to defaults
                    if 'school_input' in st.session_state:
                        st.session_state.school_input = "North High"
                    if 'grade_input' in st.session_state:
                        st.session_state.grade_input = 9
                    if 'gender_input' in st.session_state:
                        st.session_state.gender_input = "Male"
                    if 'meal_code_input' in st.session_state:
                        st.session_state.meal_code_input = "Free"
                    if 'present_days_input' in st.session_state:
                        st.session_state.present_days_input = 150
                    if 'absent_days_input' in st.session_state:
                        st.session_state.absent_days_input = 10
                    if 'academic_perf_input' in st.session_state:
                        st.session_state.academic_perf_input = 70
                    
                    # Reset the current prediction
                    if 'current_prediction' in st.session_state:
                        st.session_state.current_prediction = None
                    if 'calculation_complete' in st.session_state:
                        st.session_state.calculation_complete = False
            
            # Use a new unique text input field for new students
            new_student_id = st.text_input(
                "Enter New Student ID",
                key="new_student_id_input",
                help="Enter a unique ID for the new student"
            )
            
            # Update the student_id_input when new_student_id_input changes
            if new_student_id:
                st.session_state.student_id_input = new_student_id
            else:
                # If no text entered, ensure student_id_input exists but is blank
                if 'student_id_input' not in st.session_state:
                    st.session_state.student_id_input = ""
        else:
            # Use the selected student ID and trigger reload of student data
            if 'student_id_input' not in st.session_state or st.session_state.student_id_input != student_select:
                st.session_state.student_id_input = student_select
                # Force immediate update
                st.session_state.student_id_changed = True
            
        # Handle student ID changes using session state variables
        if 'student_id_input' in st.session_state:
            if 'prev_student_id' not in st.session_state:
                st.session_state.prev_student_id = st.session_state.student_id_input
            elif st.session_state.prev_student_id != st.session_state.student_id_input:
                # Update previous ID and trigger change
                prev_id = st.session_state.prev_student_id
                current_id = st.session_state.student_id_input
                st.session_state.prev_student_id = current_id
                
                # Don't call functions directly with on_change - use a flag in session state instead
                if 'student_id_changed' not in st.session_state:
                    st.session_state.student_id_changed = True
        
        # Process student ID change if needed
        if 'student_id_changed' in st.session_state and st.session_state.student_id_changed:
            on_student_id_change()
            st.session_state.student_id_changed = False
        
        # Create a form for student data inputs
        with st.form(key="ca_input_form", clear_on_submit=False):
            # Create a layout for student details
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                # School input - use session state value if available
                school_options = ["North High", "South High", "East Middle", "West Elementary", "Central Academy"]
                
                # Set a default index if session state value isn't in the options
                if 'school_input' in st.session_state and st.session_state.school_input in school_options:
                    default_idx = school_options.index(st.session_state.school_input)
                else:
                    default_idx = 0
                    # Reset session state to avoid errors
                    if 'school_input' in st.session_state and st.session_state.school_input not in school_options:
                        st.session_state.school_input = school_options[0]
                
                school = st.selectbox(
                    "School",
                    options=school_options,
                    index=default_idx,
                    key="school_input"
                )
                
                # Grade input
                grade = st.number_input(
                    "Grade",
                    min_value=1,
                    max_value=12,
                    value=9,
                    key="grade_input"
                )
                
                # Gender input
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    key="gender_input"
                )
                
                # Meal code
                meal_code = st.selectbox(
                    "Meal Code",
                    options=["Free", "Reduced", "Paid"],
                    key="meal_code_input"
                )
            
            with details_col2:
                # Attendance details
                present_days = st.number_input(
                    "Present Days",
                    min_value=0,
                    max_value=200,
                    value=150,
                    key="present_days_input"
                )
                
                absent_days = st.number_input(
                    "Absent Days",
                    min_value=0,
                    max_value=200,
                    value=10,
                    key="absent_days_input"
                )
                
                # Calculate attendance percentage
                total_days = present_days + absent_days
                attendance_pct = (present_days / total_days * 100) if total_days > 0 else 0
                
                st.metric("Attendance Percentage", f"{attendance_pct:.1f}%")
                
                # Academic performance
                academic_perf = st.slider(
                    "Academic Performance",
                    min_value=0,
                    max_value=100,
                    value=70,
                    key="academic_perf_input"
                )
            
            # Submit button
            submit_button = st.form_submit_button(label="Calculate CA Risk", on_click=on_calculate_risk)
    
    with col2:  # Results column
        st.markdown("<div class='card-subtitle'>🔍 Risk Assessment</div>", unsafe_allow_html=True)
        
        # Display prediction results
        if st.session_state.current_prediction is not None:
            risk_value = st.session_state.current_prediction
            
            # Display the risk gauge
            risk_fig = plot_risk_gauge(risk_value)
            
            if risk_fig:
                st.plotly_chart(risk_fig, use_container_width=True, key="risk_gauge_chart")
            
            # Display recommendation
            st.markdown("### Recommended Actions")
            recommendations = get_recommendation(risk_value)
            
            st.markdown(f"<div class='recommendation'>", unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"- {rec}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Enter student details and click 'Calculate CA Risk' to see prediction.")
    
    # What-if analysis section
    if st.session_state.calculation_complete:
        st.markdown("<div class='card-subtitle'>🔮 What-If Analysis</div>", unsafe_allow_html=True)
        
        # Place the details in an expander for cleaner interface
        with st.expander("About What-If Analysis", expanded=True):
            st.markdown("""
            Adjust the parameters below to see how changes might affect this student's CA risk.
            This can help plan interventions and understand key risk factors.
            """)
        
        whatif_col1, whatif_col2, whatif_col3 = st.columns(3)
        
        with whatif_col1:
            whatif_present = st.number_input(
                "What-If Present Days",
                min_value=0,
                max_value=200,
                value=int(st.session_state.present_days_input),
                key="what_if_present_days"
            )
        
        with whatif_col2:
            whatif_absent = st.number_input(
                "What-If Absent Days",
                min_value=0,
                max_value=200,
                value=int(st.session_state.absent_days_input),
                key="what_if_absent_days"
            )
        
        with whatif_col3:
            whatif_academic = st.number_input(
                "What-If Academic Performance",
                min_value=0,
                max_value=100,
                value=int(st.session_state.academic_perf_input),
                key="what_if_academic_perf"
            )
        
        # Calculate and show what-if prediction
        whatif_button = st.button("Calculate What-If Scenario")
        
        # Handle what-if calculation when button is clicked
        if whatif_button:
            on_calculate_what_if()
        
        # Display what-if results
        if 'what_if_prediction' in st.session_state and st.session_state.what_if_prediction is not None:
            whatif_col1, whatif_col2 = st.columns(2)
            
            with whatif_col1:
                st.markdown("#### Original Prediction")
                original_fig = plot_risk_gauge(st.session_state.original_prediction, key="original_gauge")
                if original_fig:
                    st.plotly_chart(original_fig, use_container_width=True, key="orig_fig_chart")
            
            with whatif_col2:
                st.markdown("#### What-If Prediction")
                whatif_fig = plot_risk_gauge(st.session_state.what_if_prediction, key="whatif_gauge")
                if whatif_fig:
                    st.plotly_chart(whatif_fig, use_container_width=True, key="whatif_fig_chart")
            
            # Show difference
            risk_diff = st.session_state.what_if_prediction - st.session_state.original_prediction
            diff_text = "increased" if risk_diff > 0 else "decreased"
            
            st.markdown(f"""
            <div class='info-box'>
            The what-if scenario has <b>{diff_text}</b> the risk by <b>{abs(risk_diff)*100:.1f}%</b>.
            </div>
            """, unsafe_allow_html=True)
            
            # Show what-if recommendations
            st.markdown("#### What-If Recommendations")
            whatif_recommendations = get_recommendation(st.session_state.what_if_prediction, what_if=True)
            
            st.markdown(f"<div class='recommendation'>", unsafe_allow_html=True)
            for rec in whatif_recommendations:
                st.markdown(f"- {rec}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main application
def main():
    """Main application entry point"""
    # Initialize the session state
    initialize_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render the sidebar and get the selected mode
    app_mode = render_sidebar()
    
    # Header
    st.markdown("<h1 class='main-header'>Chronic Absenteeism Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Identify at-risk students and plan effective interventions</p>", unsafe_allow_html=True)
    
    # Render the selected section
    if app_mode == "System Training":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='icon-header'><span class='emoji'>🧠</span><h2>System Training</h2></div>", unsafe_allow_html=True)
        st.markdown("""
        Train the prediction model using historical student data. The system will learn patterns 
        that lead to chronic absenteeism and use these to predict future risk.
        """)
        
        # Training section tabs
        training_tabs = st.tabs(["Training Data", "Model Parameters", "Training Results"])
        
        with training_tabs[0]:  # Training Data tab
            render_training_data_tab()
        
        with training_tabs[1]:  # Model Parameters tab
            render_model_params_tab()
        
        with training_tabs[2]:  # Results tab
            render_training_results_tab()
        
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif app_mode == "Batch Prediction":
        render_batch_prediction()
        
        # Individual student prediction section
        render_individual_prediction()
            
    elif app_mode == "Advanced Analytics":
        render_advanced_analytics()
            
    elif app_mode == "System Settings":
        render_system_settings()

# Run the application
if __name__ == "__main__":
    main()