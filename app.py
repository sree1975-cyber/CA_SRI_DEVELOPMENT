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
def get_risk_explanation(risk_value, student_data):
    """Generate clear explanation of risk factors"""
    explanations = []
    
    if risk_value >= 0.7:
        explanations.append("🚨 **Critical Risk Level** (70%+ probability of chronic absenteeism)")
    elif risk_value >= 0.4:
        explanations.append("⚠️ **Elevated Risk Level** (40-69% probability)")
    else:
        explanations.append("✅ **Low Risk Level** (Good attendance patterns)")
    
    # Attendance factors
    attendance_pct = (student_data.get('Present_Days', 0) / 
                    (student_data.get('Present_Days', 0) + student_data.get('Absent_Days', 1)) * 100
    if attendance_pct < 85:
        explanations.append(f"• Low attendance rate ({attendance_pct:.1f}%)")
    
    # Academic factors
    if student_data.get('Academic_Perf', 100) < 65:
        explanations.append(f"• Below-average academics ({student_data.get('Academic_Perf')}%)")
    
    # Socioeconomic factors
    if student_data.get('Meal_Code', '') in ['Free', 'Reduced']:
        explanations.append("• Eligible for meal assistance (potential socioeconomic factors)")
    
    return "\n".join(explanations)

def render_individual_prediction():
    """Render the prediction interface with auto-updating fields"""
    # UI Setup
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>👨‍🎓 Student Risk Analysis</div>", unsafe_allow_html=True)
    
    # Data Validation
    if 'current_year_data' not in st.session_state:
        st.error("No student data loaded. Please upload current-year data first.")
        return
    
    # Get current students
    current_students = st.session_state.current_year_data['Student_ID'].dropna().unique().tolist()
    if not current_students:
        st.error("No valid student IDs found.")
        return
    
    # Student Selection
    selected_id = st.selectbox(
        "Select Student",
        options=current_students,
        index=0,
        key="student_select"
    )
    
    # Get CURRENT student data (auto-updates on selection change)
    current_student = st.session_state.current_year_data[
        st.session_state.current_year_data['Student_ID'] == selected_id
    ].iloc[0].to_dict()
    
    # Auto-updating Form
    with st.form(key="student_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # School (with safe default)
            school = st.selectbox(
                "School",
                options=["North High", "South High", "East Middle", "West Elementary", "Central Academy"],
                index=0,
                key=f"school_{selected_id}"  # Unique key per student
            )
            
            # Grade
            grade = st.number_input(
                "Grade",
                min_value=1,
                max_value=12,
                value=int(current_student.get('Grade', 9)),
                key=f"grade_{selected_id}"
            )
            
            # Attendance
            present_days = st.number_input(
                "Present Days",
                min_value=0,
                max_value=200,
                value=int(current_student.get('Present_Days', 150)),
                key=f"present_{selected_id}"
            )
            
            absent_days = st.number_input(
                "Absent Days",
                min_value=0,
                max_value=200,
                value=int(current_student.get('Absent_Days', 10)),
                key=f"absent_{selected_id}"
            )
            
            # Auto-calculated attendance
            total_days = present_days + absent_days
            st.metric(
                "Attendance Rate", 
                f"{(present_days/total_days*100 if total_days>0 else 0):.1f}%"
            )
        
        with col2:
            # Academic Performance (NOW AUTO-UPDATING)
            academic_perf = st.slider(
                "Academic Performance %",
                min_value=0,
                max_value=100,
                value=int(current_student.get('Academic_Perf', 70)),
                key=f"academic_{selected_id}"
            )
            
            # Demographic factors
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female", "Other"],
                index=["Male", "Female", "Other"].index(current_student.get('Gender', 'Male')),
                key=f"gender_{selected_id}"
            )
            
            meal_code = st.selectbox(
                "Meal Status",
                options=["Free", "Reduced", "Paid"],
                index=["Free", "Reduced", "Paid"].index(current_student.get('Meal_Code', 'Free')),
                key=f"meal_{selected_id}"
            )
        
        # Submit button
        if st.form_submit_button("Analyze Risk"):
            on_calculate_risk()  # Your existing prediction function
    
    # Results Display
    if st.session_state.get('current_prediction'):
        risk_value = st.session_state.current_prediction
        
        # Risk Gauge
        st.plotly_chart(plot_risk_gauge(risk_value), use_container_width=True)
        
        # Clear Risk Explanation
        st.markdown("### Risk Analysis")
        st.markdown(get_risk_explanation(risk_value, current_student))
        
        # Interventions
        st.markdown("### Recommended Actions")
        for intervention, reason in get_recommendation_with_reasons(risk_value, current_student):
            st.markdown(f"""
            <div style="padding:10px; margin:8px 0; border-left:4px solid #4CAF50; background:#f8f9fa;">
                <div style="font-weight:bold; font-size:15px;">{intervention}</div>
                <div style="color:#555; font-size:13px;">{reason}</div>
            </div>
            """, unsafe_allow_html=True)
    
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
