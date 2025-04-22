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
        st.markdown("Upload a CSV file with current student data to predict CA risk.")
        
        with st.expander("CSV File Format Details"):
            st.markdown("""
            The CSV file should include:
            - Student_ID (optional)
            - School
            - Grade
            - Gender
            - Meal_Code
            - Academic_Performance
            - Year
            - Present_Days
            - Absent_Days
            - Attendance_Percentage
            """)
        
        current_data = upload_data_file(file_type="current")
        
        if 'current_year_data' in st.session_state and not st.session_state.current_year_data.empty:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.current_year_data.head(5), use_container_width=True)
            
            if st.button("Generate Predictions", key="batch_predict_button"):
                with st.spinner("Generating predictions..."):
                    try:
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
        
        if 'prediction_results' not in st.session_state or st.session_state.prediction_results is None:
            st.info("No prediction results available. Please upload data and run prediction first.")
        
        if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None:
            results = st.session_state.prediction_results
            
            # Risk summary
            high_risk_count = len(results[results['Risk_Category'] == 'High'])
            medium_risk_count = len(results[results['Risk_Category'] == 'Medium'])
            low_risk_count = len(results[results['Risk_Category'] == 'Low'])
            
            summary_df = pd.DataFrame({
                'Risk Category': ['High', 'Medium', 'Low'],
                'Count': [high_risk_count, medium_risk_count, low_risk_count]
            })
            
            total_students = len(results)
            summary_df['Percentage'] = (summary_df['Count'] / total_students * 100).round(1)
            
            # Metrics display
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("High Risk Students", f"{high_risk_count} ({summary_df.iloc[0]['Percentage']}%)")
            with metrics_col2:
                st.metric("Medium Risk Students", f"{medium_risk_count} ({summary_df.iloc[1]['Percentage']}%)")
            with metrics_col3:
                st.metric("Low Risk Students", f"{low_risk_count} ({summary_df.iloc[2]['Percentage']}%)")
            
            # Pie chart
            fig = px.pie(
                summary_df,
                values='Count',
                names='Risk Category',
                title='Risk Distribution',
                color='Risk Category',
                color_discrete_map={'High': 'red', 'Medium': 'gold', 'Low': 'green'}
            )
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Filters
            st.markdown("### Filter and Search")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                available_schools = results['School'].unique().tolist() if 'School' in results.columns else []
                selected_school = st.selectbox(
                    "School",
                    options=["All"] + available_schools,
                    key="results_school_filter"
                )
            
            with filter_col2:
                selected_risk = st.multiselect(
                    "Risk Category",
                    options=['High', 'Medium', 'Low'],
                    default=['High'],
                    key="results_risk_filter"
                )
            
            with filter_col3:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                reset_filters_button = st.button("Reset Filters", key="reset_filters_button")
            
            def apply_filters(df):
                filtered_df = df.copy()
                if selected_school != "All" and 'School' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['School'] == selected_school]
                if selected_risk:
                    filtered_df = filtered_df[filtered_df['Risk_Category'].isin(selected_risk)]
                return filtered_df
            
            filtered_results = apply_filters(results)
            
            # Highlighting function
            def highlight_risk(val):
                if pd.isna(val):
                    return 'background-color: #FFFFFF'
                if isinstance(val, str):
                    if val == 'High':
                        return 'background-color: #FFCCCC'
                    elif val == 'Medium':
                        return 'background-color: #FFFFCC'
                    elif val == 'Low':
                        return 'background-color: #CCFFCC'
                try:
                    val_float = float(val)
                    if val_float >= 0.7:
                        return 'background-color: #FFCCCC'
                    elif val_float >= 0.3:
                        return 'background-color: #FFFFCC'
                    else:
                        return 'background-color: #CCFFCC'
                except (ValueError, TypeError):
                    return 'background-color: #FFFFFF'
            
            # Display filtered results
            display_cols = ['Student_ID', 'School', 'Grade', 'Gender', 'CA_Risk', 'Risk_Category']
            display_cols = [col for col in display_cols if col in filtered_results.columns]
            display_df = filtered_results[display_cols].copy()
            
            styled_results = display_df.style.map(
                highlight_risk, subset=['Risk_Category']
            )
            if 'CA_Risk' in display_df.columns:
                styled_results = styled_results.map(
                    highlight_risk, subset=['CA_Risk']
                )
            
            st.dataframe(styled_results, use_container_width=True)
            
            # Export section
            st.markdown("### Export Results")
            export_col1, export_col2, export_col3, export_col4 = st.columns([2, 1, 1, 1])
            
            with export_col1:
                export_cols = st.multiselect(
                    "Columns to Export",
                    options=results.columns.tolist(),
                    default=['Student_ID', 'School', 'Grade', 'Gender', 'CA_Risk', 'Risk_Category'],
                    key="export_columns"
                )
            
            with export_col2:
                export_risk = st.multiselect(
                    "Risk Levels",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium', 'Low'],
                    key="export_risk"
                )
            
            with export_col4:
                export_button = st.button("Export CSV", key="export_csv_button")
            
            if export_button:
                if not export_cols:
                    st.warning("Please select at least one column to export.")
                else:
                    export_data = results[results['Risk_Category'].isin(export_risk)]
                    export_data = export_data[export_cols]
                    csv = export_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="ca_prediction_results.csv",
                        mime="text/csv"
                    )
            
            # Individual Student Analysis
            st.markdown("---")
            st.markdown("### Individual Student Analysis (Predicted Results)")
            
            if not filtered_results.empty:
                student_list = filtered_results['Student_ID'].tolist()
                selected_student_id = st.selectbox(
                    "Select Student for Detailed Analysis",
                    options=student_list,
                    key="batch_student_select"
                )
                
                if selected_student_id:
                    student_data = filtered_results[filtered_results['Student_ID'] == selected_student_id].iloc[0]
                    risk_value = student_data['CA_Risk']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.plotly_chart(
                            plot_risk_gauge(risk_value),
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )
                    
                    with col2:
                        st.markdown("#### Risk Analysis (Predicted)")
                        explanation = get_recommendation(risk_value, student_data.to_dict())
                        st.markdown(explanation)
                        
                        st.markdown("#### Recommended Interventions")
                        recommendations = get_recommendations(risk_value, student_data.to_dict())
                        for intervention, reason in recommendations:
                            st.markdown(f"""
                            <div style="padding:10px; margin:10px 0; background:#f8f9fa; 
                                        border-left:4px solid #4CAF50; border-radius:4px;">
                                <div style="font-weight:500; color:#333;">{intervention}</div>
                                <div style="font-size:0.9em; color:#666;">{reason}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with st.expander("View Detailed Student Data"):
                        profile_cols = ['Student_ID', 'School', 'Grade', 'Gender', 'Meal_Code',
                                      'Academic_Performance', 'Present_Days', 'Absent_Days',
                                      'CA_Risk', 'Risk_Category']
                        profile_data = student_data[[col for col in profile_cols if col in student_data.index]]
                        st.table(profile_data.astype(str))

    st.markdown("</div>", unsafe_allow_html=True)
    return


def display_svg(file_path, width=None):
    """Display an SVG file in a Streamlit app"""
    import os
    
    if not os.path.exists(file_path):
        return f"<div style='text-align: center; color: #888;'>Image not found: {file_path}</div>"
        
    with open(file_path, "r") as f:
        content = f.read()
        
    if width:
        content = content.replace("<svg ", f"<svg width='{width}' ")
        
    return content
