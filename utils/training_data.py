"""
Training Data tab functionality
"""

import streamlit as st
import pandas as pd
from utils.common import upload_data_file

def render_training_data_tab():
    """Render the Training Data tab content"""
    st.markdown("<div class='card-title'>📊 Training Data</div>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    
    # Action buttons row
    action_col1, action_col2 = st.columns([1, 4])
    
    with action_col1:
        # Clear Training Data button to reset just the training process
        if st.button("🔄 Clear Training Data", key="clear_training_button"):
            # Reset session state for training only
            if 'historical_data' in st.session_state:
                del st.session_state.historical_data
            if 'training_data_processed' in st.session_state:
                del st.session_state.training_data_processed
            if 'training_results' in st.session_state:
                del st.session_state.training_results
            if 'trained_model' in st.session_state:
                del st.session_state.trained_model
            
            st.success("Training data cleared. Please upload new training data.")
            st.rerun()  # Use st.rerun() instead of experimental_rerun()
    
    # Data upload section with cleaner UI
    st.markdown("### Upload Historical Student Data")
    
    # Put detailed instructions in an expander
    with st.expander("CSV Format Instructions"):
        st.markdown("""
        Please upload a CSV file containing historical student data with the following columns:
        - `Student_ID` (optional): Unique identifier for each student
        - `School`: School name
        - `Grade`: Student's grade level
        - `Gender`: Student's gender
        - `Meal_Code`: Free, Reduced, or Paid
        - `Academic_Performance`: Score (0-100)
        - `Year`: School year
        - `Present_Days`: Number of days present
        - `Absent_Days`: Number of days absent
        - `Attendance_Percentage`: Percentage of attendance
        - `CA_Status`: Chronic absenteeism status (can be either 1/0 or "CA"/"NO_CA")
        """)
    
    # File uploader for historical data
    uploaded_data = upload_data_file(file_type="historical")
    
    # Data preview if available
    if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
        st.markdown("### Data Preview")
        
        # Show data statistics
        data_stats = st.session_state.historical_data.describe(include='all').T
        
        # Create two columns for positive and negative examples
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            # Count records
            st.metric("Total Records", len(st.session_state.historical_data))
            
            # Show student statistics
            num_students = st.session_state.historical_data['Student_ID'].nunique()
            st.metric("Unique Students", num_students)
        
        with stats_col2:
            # CA distribution if available
            if 'CA_Status' in st.session_state.historical_data.columns:
                # Handle both numeric (0/1) and text ('CA'/'NO_CA') formats
                if st.session_state.historical_data['CA_Status'].dtype == 'object':
                    ca_count = (st.session_state.historical_data['CA_Status'] == 'CA').sum()
                else:
                    ca_count = st.session_state.historical_data['CA_Status'].sum()
                
                non_ca_count = len(st.session_state.historical_data) - ca_count
                ca_percent = (ca_count / len(st.session_state.historical_data)) * 100
                
                st.metric("CA Students", f"{ca_count} ({ca_percent:.1f}%)")
                st.metric("Non-CA Students", non_ca_count)
        
        # Data preview
        st.dataframe(st.session_state.historical_data.head(10), use_container_width=True)
        
        # Data validation alerts
        validation_issues = []
        
        # Check for missing required columns (Student_ID is optional now)
        required_cols = ['CA_Status', 'School', 'Grade', 'Year', 'Present_Days', 'Absent_Days']
        missing_cols = [col for col in required_cols if col not in st.session_state.historical_data.columns]
        if missing_cols:
            validation_issues.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check for missing values in key columns
        if not missing_cols:
            for col in required_cols:
                missing_count = st.session_state.historical_data[col].isna().sum()
                if missing_count > 0:
                    validation_issues.append(f"Column '{col}' has {missing_count} missing values")
        
        # Display validation issues if any
        if validation_issues:
            st.warning("Data Validation Issues:")
            for issue in validation_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ Training data validation passed. Ready for model training.")
            
            # Add guidance to direct user to the next step
            st.info("👉 Now go to the **Model Parameters** tab to configure and train the model.")
    
    # Feature explanation
    with st.expander("Feature Descriptions & Guidelines"):
        st.markdown("""
        ### Key Features for Chronic Absenteeism Prediction
        
        | Feature | Description | Data Type | Example Values |
        | ------- | ----------- | --------- | -------------- |
        | Student_ID | Unique identifier for each student (optional) | String | "ST001", "2023045" |
        | School | Name of the school | Categorical | "North High", "Central Elementary" |
        | Grade | Student's grade level | Numeric (6-12) | 9, 10, 11 |
        | Gender | Student's gender | Categorical | "Male", "Female" |
        | Present_Days | Days present during period | Numeric | 150, 160, 170 |
        | Absent_Days | Days absent during period | Numeric | 15, 20, 25 |
        | Attendance_Percentage | Percentage of days attended | Numeric (0-100) | 85.5, 90.2, 95.0 |
        | Meal_Code | Student's meal subsidy status | Categorical | "Free", "Reduced", "Paid" |
        | Academic_Performance | Academic performance score | Numeric (0-100) | 75, 82, 90 |
        | Year | School year | Numeric | 2023, 2024 |
        | CA_Status | Whether student was chronically absent | Binary (0/1) or Text ("CA"/"NO_CA") | 1, 0, "CA", "NO_CA" |
        
        **Note:** The system will automatically calculate any derived features during preprocessing.
        """)
    
    return