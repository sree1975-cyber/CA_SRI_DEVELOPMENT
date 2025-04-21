"""
Batch Prediction functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.common import upload_data_file, batch_predict_ca, predict_ca_risk, plot_risk_gauge, get_recommendation

def render_batch_prediction():
    """Render the Batch Prediction section"""
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(display_svg("images/batch_prediction.svg", width="200px"), unsafe_allow_html=True)
    st.markdown("<h2>Batch Prediction</h2>", unsafe_allow_html=True)
    st.markdown("""
    Upload current student data to predict chronic absenteeism risk for multiple students at once.
    """)
    
    # Check if we have a trained model
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No trained model available. Please train a model in the System Training section first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Batch prediction tabs
    batch_tabs = st.tabs(["Data Upload", "Prediction Results"])
    
    with batch_tabs[0]:  # Data Upload tab
        st.markdown("<div class='card-title'>📤 Upload Current Year Data</div>", unsafe_allow_html=True)
        
        # Data upload section with expandable details
        # Simple instruction first
        st.markdown("Upload a CSV file with current student data to predict CA risk.")
        
        # Detailed information in an expander
        with st.expander("CSV File Format Details"):
            st.markdown("""
            The CSV file should include the following information:
            - Student_ID (optional): Unique identifier for each student
            - School: School name
            - Grade: Student's grade level
            - Gender: Student's gender
            - Meal_Code: Free, Reduced, or Paid
            - Academic_Performance: Score (0-100)
            - Year: School year (current year)
            - Present_Days: Number of days present
            - Absent_Days: Number of days absent
            - Attendance_Percentage: Percentage of attendance
            """)
        
        # File uploader for current student data
        current_data = upload_data_file(file_type="current")
        
        # If data is available, enable prediction
        if 'current_year_data' in st.session_state and not st.session_state.current_year_data.empty:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.current_year_data.head(5), use_container_width=True)
            
            # Run prediction button
            if st.button("Generate Predictions", key="batch_predict_button"):
                with st.spinner("Generating predictions..."):
                    # Run the prediction
                    try:
                        # Execute prediction with better error handling
                        predictions = batch_predict_ca(st.session_state.current_year_data, st.session_state.model)
                        
                        if predictions is not None and not predictions.empty:
                            st.session_state.prediction_results = predictions
                            st.session_state.prediction_complete = True
                            st.success("✅ Predictions generated successfully! Go to Prediction Results tab to view.")
                        else:
                            st.error("Error generating predictions. Please check your data.")
                    except Exception as e:
                        st.error(f"Error in prediction pre-processing: {str(e)}")
                        st.warning("If you're seeing an array truth value error, this often happens due to data type issues. Try checking your input data format.")
    
    with batch_tabs[1]:  # Prediction Results tab
        st.markdown("<div class='card-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
        
        # Check if predictions are available
        if 'prediction_results' not in st.session_state or st.session_state.prediction_results is None:
            st.info("No prediction results available. Please upload data and run prediction first.")
            # Don't return here - we want to show the rest of the UI
        
        # Display prediction results
        st.markdown("### CA Risk Predictions")
        
        # Only proceed with visualization if results are available
        if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None:
            # Get the prediction results
            results = st.session_state.prediction_results
            
            # Display a summary of the results
            high_risk_count = len(results[results['Risk_Category'] == 'High'])
            medium_risk_count = len(results[results['Risk_Category'] == 'Medium'])
            low_risk_count = len(results[results['Risk_Category'] == 'Low'])
            
            # Create a DataFrame for the summary
            summary_df = pd.DataFrame({
                'Risk Category': ['High', 'Medium', 'Low'],
                'Count': [high_risk_count, medium_risk_count, low_risk_count]
            })
            
            # Calculate percentages
            total_students = len(results)
            summary_df['Percentage'] = (summary_df['Count'] / total_students * 100).round(1)
            
            # Display metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("High Risk Students", f"{high_risk_count} ({summary_df.iloc[0]['Percentage']}%)")
            with metrics_col2:
                st.metric("Medium Risk Students", f"{medium_risk_count} ({summary_df.iloc[1]['Percentage']}%)")
            with metrics_col3:
                st.metric("Low Risk Students", f"{low_risk_count} ({summary_df.iloc[2]['Percentage']}%)")
            
            # Create a pie chart for risk distribution
            fig = px.pie(
                summary_df,
                values='Count',
                names='Risk Category',
                title='Risk Distribution',
                color='Risk Category',
                color_discrete_map={'High': 'red', 'Medium': 'gold', 'Low': 'green'}
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Filter and search options
            st.markdown("### Filter and Search")
            
            # Create a layout for filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # School filter
                available_schools = results['School'].unique().tolist() if 'School' in results.columns else []
                selected_school = st.selectbox(
                    "School",
                    options=["All"] + available_schools,
                    key="results_school_filter"
                )
            
            with filter_col2:
                # Risk category filter
                selected_risk = st.multiselect(
                    "Risk Category",
                    options=['High', 'Medium', 'Low'],
                    default=['High'],
                    key="results_risk_filter"
                )
            
            with filter_col3:
                # Reset button
                st.markdown("&nbsp;", unsafe_allow_html=True)  # Add some space
                reset_filters_button = st.button("Reset Filters", key="reset_filters_button")
            
            # Define filters function
            def apply_filters(df):
                filtered_df = df.copy()
                
                # Apply school filter
                if selected_school != "All" and 'School' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['School'] == selected_school]
                
                # Apply risk filter
                if selected_risk:
                    filtered_df = filtered_df[filtered_df['Risk_Category'].isin(selected_risk)]
                
                return filtered_df
            
            # Handle reset button using callbacks for proper state management
            def reset_filters():
                if 'results_school_filter' in st.session_state:
                    del st.session_state.results_school_filter
                if 'results_risk_filter' in st.session_state:
                    del st.session_state.results_risk_filter
                
            # Create reset button with callback
            reset_filters_button = st.button("Reset Filters", 
                                            key="reset_filters_button_batch", 
                                            on_click=reset_filters)
            
            # Apply filters
            filtered_results = apply_filters(results)
            
            # Create a form to ensure filter submission button
            with st.form(key="risk_filtering_form"):
                st.markdown(f"### Students at Risk ({len(filtered_results)} of {len(results)})")
                # Submit button for the form
                filter_submit = st.form_submit_button("Apply Filters")
            
            # Add highlighting for risk levels - updated to use the recommended Styler.map method
            def highlight_risk(val):
                if pd.isna(val):
                    return 'background-color: #FFFFFF'  # White for missing data
                
                if isinstance(val, str):
                    if val == 'High':
                        return 'background-color: #FFCCCC'  # Light red
                    elif val == 'Medium':
                        return 'background-color: #FFFFCC'  # Light yellow
                    elif val == 'Low':
                        return 'background-color: #CCFFCC'  # Light green
                    
                # Try to convert numeric values
                try:
                    val_float = float(val)
                    if val_float >= 0.7:
                        return 'background-color: #FFCCCC'  # Light red
                    elif val_float >= 0.3:
                        return 'background-color: #FFFFCC'  # Light yellow
                    else:
                        return 'background-color: #CCFFCC'  # Light green
                except (ValueError, TypeError):
                    pass
                
                return 'background-color: #FFFFFF'  # Default white
            
            # Select columns to display
            display_cols = ['Student_ID', 'School', 'Grade', 'Gender', 'CA_Risk', 'Risk_Category']
            display_cols = [col for col in display_cols if col in filtered_results.columns]
            
            # Create a dataframe for display
            display_df = filtered_results[display_cols].copy()
            
            # Apply highlighting using the newer pandas Styler.map method (replacing deprecated applymap)
            styled_results = display_df.style.map(
                highlight_risk, subset=['Risk_Category']
            )
            
            # Also highlight the risk value column
            if 'CA_Risk' in display_df.columns:
                styled_results = styled_results.map(
                    highlight_risk, subset=['CA_Risk']
                )
            
            # Display the table
            st.dataframe(styled_results, use_container_width=True)
            
            # Export options
            st.markdown("### Export Results")

            
            # Add individual student analysis section after export
    st.markdown("---")
    st.markdown("### Individual Student Analysis")
    
    if not filtered_results.empty:
        # Student selection
        student_list = filtered_results['Student_ID'].tolist()
        selected_student_id = st.selectbox(
            "Select Student for Detailed Analysis",
            options=student_list,
            key="batch_student_select"
        )
        
        if selected_student_id:
            # Get student data
            student_data = filtered_results[filtered_results['Student_ID'] == selected_student_id].iloc[0]
            risk_value = student_data['CA_Risk']
            
            # Create layout columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display interactive risk gauge
                st.plotly_chart(
                    plot_risk_gauge(risk_value),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )
            
            with col2:
                # Risk explanation
                st.markdown("#### Risk Factors Analysis")
                explanation = get_risk_explanation(risk_value, student_data)
                st.markdown(explanation)
                
                # Recommendations
                st.markdown("#### Recommended Interventions")
                recommendations = get_recommendation_with_reasons(risk_value, student_data)
                for intervention, reason in recommendations:
                    st.markdown(f"""
                    <div style="padding:10px; margin:10px 0; background:#f8f9fa; 
                                border-left:4px solid #4CAF50; border-radius:4px;">
                        <div style="font-weight:500; color:#333;">{intervention}</div>
                        <div style="font-size:0.9em; color:#666;">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Show raw student data
            with st.expander("View Detailed Student Data"):
                st.write("### Student Profile")
                profile_cols = ['School', 'Grade', 'Gender', 'Meal_Code',
                               'Academic_Performance', 'Present_Days', 'Absent_Days']
                profile_data = student_data[profile_cols]
                st.table(profile_data.astype(str))
            
            
            # Create a layout for export options
            export_col1, export_col2, export_col3, export_col4 = st.columns([2, 1, 1, 1])
            
            with export_col1:
                # Select columns to export
                export_cols = st.multiselect(
                    "Columns to Export",
                    options=results.columns.tolist(),
                    default=['Student_ID', 'School', 'Grade', 'Gender', 'CA_Risk', 'Risk_Category'],
                    key="export_columns"
                )
            
            with export_col2:
                # Risk filter for export
                export_risk = st.multiselect(
                    "Risk Levels",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium', 'Low'],
                    key="export_risk"
                )
            
            with export_col3:
                # Reset export filters - just placeholder now
                st.markdown("&nbsp;", unsafe_allow_html=True)  # Add some space
            
            with export_col4:
                # Export button
                st.markdown("&nbsp;", unsafe_allow_html=True)  # Add some space
                export_button = st.button("Export CSV", key="export_csv_button")
            
            # Handle reset export filters using callback
            def reset_export_filters():
                if 'export_columns' in st.session_state:
                    del st.session_state.export_columns
                if 'export_risk' in st.session_state:
                    del st.session_state.export_risk
                
            # Create reset button with callback
            reset_export_button = st.button("Reset", 
                                          key="reset_export_button", 
                                          on_click=reset_export_filters)
            
            # Handle export
            if export_button:
                if not export_cols:
                    st.warning("Please select at least one column to export.")
                else:
                    # Filter by selected risk levels
                    export_data = results[results['Risk_Category'].isin(export_risk)]
                    
                    # Select columns
                    export_data = export_data[export_cols]
                    
                    # Convert to CSV
                    csv = export_data.to_csv(index=False)
                    
                    # Create a download link
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="ca_prediction_results.csv",
                        mime="text/csv"
                    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    return


def display_svg(file_path, width=None):
    """Display an SVG file in a Streamlit app"""
    import os
    
    if not os.path.exists(file_path):
        # Return a default placeholder if the file doesn't exist
        return f"<div style='text-align: center; color: #888;'>Image not found: {file_path}</div>"
        
    with open(file_path, "r") as f:
        content = f.read()
        
    if width:
        # Add width attribute to the SVG tag
        content = content.replace("<svg ", f"<svg width='{width}' ")
        
    return content
