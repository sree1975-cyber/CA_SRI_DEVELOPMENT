"""
Chronic Absenteeism Predictor - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go


# Import utility modules
from utils.common import (
    display_svg, generate_sample_data, predict_ca_risk,
    plot_risk_gauge, plot_feature_importance, get_recommendation_with_reasons,
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

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def plot_risk_gauge(risk_value):
    """Create a properly sized risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Absenteeism Risk Score", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_value*100}
        }
    ))
    fig.update_layout(
        height=300,  # Fixed height
        width=400,   # Fixed width
        margin=dict(t=50, b=10, l=50, r=50)
    )
    return fig

def get_risk_explanation(risk_value, student_data):
    """Generate clear explanation of risk factors"""
    explanations = []
    
    if risk_value >= 0.7:
        explanations.append("🚨 **Critical Risk Level** (70%+ probability of chronic absenteeism)")
    elif risk_value >= 0.4:
        explanations.append("⚠️ **Elevated Risk Level** (40-69% probability)")
    else:
        explanations.append("✅ **Low Risk Level** (Good attendance patterns)")
    
    present_days = student_data.get('Present_Days', 0)
    absent_days = student_data.get('Absent_Days', 1)
    attendance_pct = (present_days / (present_days + absent_days)) * 100
    
    if attendance_pct < 85:
        explanations.append(f"• Low attendance rate ({attendance_pct:.1f}%)")
    
    academic_performance = student_data.get('Academic_Performance', 100)
    if academic_performance < 65:
        explanations.append(f"• Below-average academics ({academic_performance}%)")
    
    if student_data.get('Meal_Code', '') in ['Free', 'Reduced']:
        explanations.append("• Eligible for meal assistance (potential socioeconomic factors)")
    
    return "\n".join(explanations)

def get_recommendation_with_reasons(risk_value, student_data):
    """Generate interventions with explanations based on risk factors"""
    interventions = []
    
    if risk_value >= 0.7:
        interventions.append((
            "🚨 Immediate 1-on-1 meeting with school counselor",
            "Student is at very high risk of chronic absenteeism"
        ))
        interventions.append((
            "📞 Parent/guardian conference within 48 hours",
            "Early family engagement is critical"
        ))
        
        if student_data.get('Absent_Days', 0) > 15:
            interventions.append((
                "🩺 Schedule health checkup",
                f"High absence days ({student_data.get('Absent_Days')})"
            ))
            
        if student_data.get('Academic_Performance', 70) < 60:
            interventions.append((
                "📚 Assign academic support tutor",
                f"Low performance ({student_data.get('Academic_Performance')}%)"
            ))

    elif risk_value >= 0.3:
        interventions.append((
            "📅 Weekly check-ins with homeroom teacher",
            "Regular monitoring prevents escalation"
        ))
        interventions.append((
            "✉️ Send personalized attendance report",
            "Family awareness improves outcomes"
        ))
        
        if student_data.get('Meal_Code', '') in ['Free', 'Reduced']:
            interventions.append((
                "🍎 Connect with nutrition programs",
                "Address potential food insecurity"
            ))

    else:
        interventions.append((
            "👍 Positive reinforcement",
            "Maintaining good patterns prevents issues"
        ))
        
        if student_data.get('Present_Days', 0) < 160:
            interventions.append((
                "🎯 Set attendance improvement goal",
                f"Current attendance: {student_data.get('Present_Days')} days"
            ))

    return interventions

def on_calculate_risk():
    """Calculate risk score based on form inputs"""
    try:
        selected_id = st.session_state.get("student_select")
        if not selected_id:
            st.error("No student selected")
            return
        
        inputs = {
            'Present_Days': st.session_state.get(f"present_{selected_id}", 150),
            'Absent_Days': st.session_state.get(f"absent_{selected_id}", 10),
            'Academic_Performance': st.session_state.get(f"academic_{selected_id}", 70),
            'Grade': st.session_state.get(f"grade_{selected_id}", 9),
            'Meal_Code': st.session_state.get(f"meal_{selected_id}", 'Free')
        }
        
        attendance_rate = inputs['Present_Days'] / (inputs['Present_Days'] + inputs['Absent_Days'])
        academic_factor = 1 - (inputs['Academic_Performance'] / 100)
        risk_score = (0.6 * (1 - attendance_rate)) + (0.4 * academic_factor)
        
        st.session_state.current_prediction = risk_score
        st.session_state.current_student_data = inputs
        
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        st.session_state.current_prediction = None

def render_individual_prediction():
    """Main prediction interface"""
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>👨‍🎓 Student Risk Analysis</div>", unsafe_allow_html=True)
    
    if 'current_year_data' not in st.session_state:
        st.error("Please upload current-year data first")
        return
    
    required_columns = ['Student_ID', 'School', 'Grade', 'Present_Days', 
                       'Absent_Days', 'Academic_Performance', 'Gender', 'Meal_Code']
    
    missing_cols = [col for col in required_columns 
                   if col not in st.session_state.current_year_data.columns]
    
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        return
    
    try:
        current_students = st.session_state.current_year_data['Student_ID'].dropna().unique().tolist()
        if not current_students:
            st.error("No valid student IDs found")
            return
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return
    
    selected_id = st.selectbox(
        "Select Student",
        options=current_students,
        index=0,
        key="student_select"
    )
    
    try:
        student_data = st.session_state.current_year_data[
            st.session_state.current_year_data['Student_ID'] == selected_id
        ].iloc[0]
    except IndexError:
        st.error("Student not found")
        return
    
    current_student = {
        'School': str(student_data.get('School', 'North High')),
        'Grade': int(student_data.get('Grade', 9)),
        'Present_Days': int(student_data.get('Present_Days', 150)),
        'Absent_Days': int(student_data.get('Absent_Days', 10)),
        'Academic_Performance': int(student_data.get('Academic_Performance', 70)),
        'Gender': str(student_data.get('Gender', 'Male')),
        'Meal_Code': str(student_data.get('Meal_Code', 'Free'))
    }
    
    with st.form(key="student_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            school_options = ["North High", "South High", "East Middle", "West Elementary", "Central Academy"]
            school_value = current_student['School']
            school_index = school_options.index(school_value) if school_value in school_options else 0
            st.selectbox(
                "School",
                options=school_options,
                index=school_index,
                key=f"school_{selected_id}"
            )
            
            st.number_input(
                "Grade",
                min_value=1,
                max_value=12,
                value=current_student['Grade'],
                key=f"grade_{selected_id}"
            )
            
            present_days = st.number_input(
                "Present Days",
                min_value=0,
                max_value=200,
                value=current_student['Present_Days'],
                key=f"present_{selected_id}"
            )
            
            absent_days = st.number_input(
                "Absent Days",
                min_value=0,
                max_value=200,
                value=current_student['Absent_Days'],
                key=f"absent_{selected_id}"
            )
            
            total_days = present_days + absent_days
            st.metric(
                "Attendance Rate", 
                f"{(present_days/total_days*100 if total_days>0 else 0):.1f}%"
            )
        
        with col2:
            st.slider(
                "Academic Performance %",
                min_value=0,
                max_value=100,
                value=current_student['Academic_Performance'],
                key=f"academic_{selected_id}"
            )
            
            gender_options = ["Male", "Female", "Other"]
            gender_value = current_student['Gender']
            gender_index = gender_options.index(gender_value) if gender_value in gender_options else 0
            st.selectbox(
                "Gender",
                options=gender_options,
                index=gender_index,
                key=f"gender_{selected_id}"
            )
            
            meal_options = ["Free", "Reduced", "Paid"]
            meal_value = current_student['Meal_Code']
            meal_index = meal_options.index(meal_value) if meal_value in meal_options else 0
            st.selectbox(
                "Meal Status",
                options=meal_options,
                index=meal_index,
                key=f"meal_{selected_id}"
            )
        
        if st.form_submit_button("Analyze Risk"):
            on_calculate_risk()
            st.rerun()
    
    if st.session_state.get('current_prediction') is not None:
        risk_value = st.session_state.current_prediction
        
        # Create columns for better layout
        col1, col2 = st.columns([1, 2])  # First column narrower for gauge
        
        with col1:
            # Display the properly sized gauge
            st.plotly_chart(
                plot_risk_gauge(risk_value),
                use_container_width=True,
                config={'displayModeBar': False}  # Hide plotly toolbar
            )
        
        with col2:
            st.markdown("### Risk Analysis")
            student_data = st.session_state.get('current_student_data', current_student)
            st.markdown(get_risk_explanation(risk_value, student_data))
            
            st.markdown("### Recommended Actions")
            for intervention, reason in get_recommendation_with_reasons(risk_value, student_data):
                st.markdown(f"""
                <div style="padding:10px; margin:8px 0; border-left:4px solid #4CAF50; background:#f8f9fa;">
                    <div style="font-weight:bold; font-size:14px;">{intervention}</div>
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
        #render_individual_prediction()
            
    elif app_mode == "Advanced Analytics":
        render_advanced_analytics()
            
    elif app_mode == "System Settings":
        render_system_settings()

# Run the application
if __name__ == "__main__":
    main()
