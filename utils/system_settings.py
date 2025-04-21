"""
System Settings functionality
"""

import streamlit as st
import os
import pandas as pd

def render_system_settings():
    """Render the System Settings section"""
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("System Settings")
    st.markdown("""
    Configure system settings and access documentation.
    """)
    
    # Settings tabs
    settings_tabs = st.tabs(["General Settings", "Data Management", "Documentation"])
    
    with settings_tabs[0]:  # General Settings tab
        st.markdown("<div class='card-title'>⚙️ General Settings</div>", unsafe_allow_html=True)
        
        # Theme settings
        st.markdown("### Theme Settings")
        
        # Theme selection
        theme = st.selectbox(
            "Color Theme",
            options=["Default", "Light", "Dark"],
            index=0
        )
        
        if theme != "Default":
            st.info(f"{theme} theme will be available in a future update.")
        
        # User information
        st.markdown("### User Information")
        
        # User details form
        with st.form("user_settings_form"):
            school_district = st.text_input("School District", "")
            admin_email = st.text_input("Administrator Email", "")
            save_settings = st.form_submit_button("Save Settings")
            
            if save_settings:
                # In a real app, these would be saved to a database or config file
                st.success("✅ Settings saved successfully!")
                
                # Store in session state for now
                st.session_state.school_district = school_district
                st.session_state.admin_email = admin_email
        
        # Data visualization settings
        st.markdown("### Visualization Settings")
        
        # Chart type preference
        chart_type = st.selectbox(
            "Preferred Chart Type",
            options=["Interactive", "Static"],
            index=0
        )
        
        if chart_type == "Static":
            st.info("Static chart option will be available in a future update.")
        
        # Date format preference
        date_format = st.selectbox(
            "Date Format",
            options=["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"],
            index=2
        )
        
        if date_format != "YYYY-MM-DD":
            st.info("Alternative date formats will be available in a future update.")
    
    with settings_tabs[1]:  # Data Management tab
        st.markdown("<div class='card-title'>🗄️ Data Management</div>", unsafe_allow_html=True)
        
        # Data storage info
        st.markdown("### Data Storage")
        
        # Display info on current data
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            st.metric("Training Data Records", len(st.session_state.historical_data))
        else:
            st.metric("Training Data Records", 0)
            
        if 'current_year_data' in st.session_state and not st.session_state.current_year_data.empty:
            st.metric("Current Year Records", len(st.session_state.current_year_data))
        else:
            st.metric("Current Year Records", 0)
            
        if 'model' in st.session_state and st.session_state.model is not None:
            st.metric("Trained Models", 1)
        else:
            st.metric("Trained Models", 0)
        
        # Data clearing options
        st.markdown("### Clear Data")
        st.markdown("""
        Use these options to clear specific data from the system. This is useful for starting fresh 
        or removing sensitive information. Note that this cannot be undone.
        """)
        
        # Create three columns for clear buttons
        clear_col1, clear_col2, clear_col3 = st.columns(3)
        
        with clear_col1:
            if st.button("Clear Training Data", key="clear_historical_data"):
                if 'historical_data' in st.session_state:
                    st.session_state.historical_data = pd.DataFrame()
                    st.success("✅ Training data cleared successfully!")
                    st.experimental_rerun()
                else:
                    st.info("No training data to clear.")
        
        with clear_col2:
            if st.button("Clear Current Year Data", key="clear_current_data"):
                if 'current_year_data' in st.session_state:
                    st.session_state.current_year_data = pd.DataFrame()
                    st.success("✅ Current year data cleared successfully!")
                    st.experimental_rerun()
                else:
                    st.info("No current year data to clear.")
        
        with clear_col3:
            if st.button("Clear Trained Model", key="clear_model"):
                if 'model' in st.session_state and st.session_state.model is not None:
                    st.session_state.model = None
                    st.session_state.training_report = None
                    st.success("✅ Trained model cleared successfully!")
                    st.experimental_rerun()
                else:
                    st.info("No trained model to clear.")
        
        # Reset all button with confirmation
        st.markdown("### Reset All Data")
        
        # Two-step confirmation to prevent accidental resets
        if 'reset_confirmation' not in st.session_state:
            st.session_state.reset_confirmation = False
        
        if not st.session_state.reset_confirmation:
            if st.button("Reset All System Data", key="reset_button"):
                st.session_state.reset_confirmation = True
                st.warning("⚠️ Are you sure? This will clear ALL data and models. This action cannot be undone.")
        else:
            confirm_col1, confirm_col2 = st.columns(2)
            
            with confirm_col1:
                if st.button("Yes, Reset Everything", key="confirm_reset_button"):
                    # Clear all session state data
                    for key in ['historical_data', 'current_year_data', 'model', 
                                'training_report', 'prediction_results', 'feature_names']:
                        if key in st.session_state:
                            if key in ['historical_data', 'current_year_data', 'prediction_results']:
                                st.session_state[key] = pd.DataFrame()
                            else:
                                st.session_state[key] = None
                    
                    st.session_state.reset_confirmation = False
                    st.success("✅ All system data has been reset!")
                    st.experimental_rerun()
            
            with confirm_col2:
                if st.button("Cancel", key="cancel_reset_button"):
                    st.session_state.reset_confirmation = False
                    st.experimental_rerun()
        
        # Data backup & restore section
        st.markdown("### Backup & Restore")
        st.markdown("""
        Export and import system data for backup or transfer to another system.
        This feature will be available in a future update.
        """)
    
    with settings_tabs[2]:  # Documentation tab
        st.markdown("<div class='card-title'>📚 Documentation</div>", unsafe_allow_html=True)
        
        # About the system
        st.markdown("### About Chronic Absenteeism Predictor")
        st.markdown("""
        The Chronic Absenteeism (CA) Predictor is a machine learning-based tool designed to help 
        educational institutions identify students at risk of chronic absenteeism. The system uses 
        historical attendance data and other student factors to predict the likelihood of a student 
        becoming chronically absent.
        
        **Version:** 1.0.0
        
        **Key Features:**
        - Historical data analysis and model training
        - Batch prediction for current students
        - Advanced analytics and risk factor analysis
        - Individual student risk assessment
        - Customizable models and parameters
        """)
        
        # User manual
        st.markdown("### User Manual")
        
        # Check if the manual file exists
        manual_path = "ca_predictor_manual.pdf"
        
        if os.path.exists(manual_path):
            with open(manual_path, "rb") as file:
                st.download_button(
                    label="Download User Manual",
                    data=file,
                    file_name="ca_predictor_manual.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("User manual file not found. Please contact system administrator.")
        
        # Technical documentation
        st.markdown("### Technical Documentation")
        st.markdown("""
        The CA Predictor uses machine learning algorithms to analyze student data and predict chronic absenteeism risk.
        
        **Model Types Available:**
        - Random Forest
        - Gradient Boosting
        - Logistic Regression
        - Neural Network (Multilayer Perceptron)
        
        **Key Risk Factors:**
        - Prior attendance patterns
        - Socioeconomic indicators
        - Academic performance
        - Transportation factors
        - School engagement
        
        **Data Requirements:**
        - Student attendance records
        - Demographic information
        - Academic records
        - Historical chronic absenteeism labels
        """)
        
        # FAQ accordion
        st.markdown("### Frequently Asked Questions")
        
        with st.expander("What is chronic absenteeism?"):
            st.markdown("""
            Chronic absenteeism is typically defined as missing 10% or more of the school year for any reason, 
            including excused and unexcused absences. For a 180-day school year, that means being absent 18 or more days.
            """)
        
        with st.expander("How accurate is the prediction model?"):
            st.markdown("""
            The accuracy of the model depends on the quality and quantity of historical data provided for training.
            In typical scenarios with good quality data, the model achieves accuracy between 80-90%. 
            You can view detailed performance metrics in the "Model Performance" tab after training.
            """)
        
        with st.expander("What data do I need to use this system?"):
            st.markdown("""
            At minimum, you need historical student data with attendance records and an indicator of whether 
            each student was chronically absent. Additional factors like academic performance, demographics, 
            and socioeconomic indicators will improve model accuracy.
            
            For prediction, you need current student data with the same features used in training.
            """)
        
        with st.expander("How do I interpret the risk levels?"):
            st.markdown("""
            - **Low Risk (0-30%)**: Students unlikely to become chronically absent; standard monitoring is sufficient
            - **Medium Risk (30-70%)**: Students who may become chronically absent; preventative measures recommended
            - **High Risk (70-100%)**: Students very likely to become chronically absent; immediate intervention required
            """)
        
        with st.expander("Can I export the predictions?"):
            st.markdown("""
            Yes, you can export prediction results as CSV files from the Batch Prediction section. 
            The export includes student information and risk assessments, which can be imported into 
            other systems or spreadsheets for further analysis or intervention planning.
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    return