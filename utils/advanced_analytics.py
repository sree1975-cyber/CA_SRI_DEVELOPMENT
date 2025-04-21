"""
Advanced Analytics functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.common import generate_system_report, plot_feature_importance

def render_advanced_analytics():
    """Render the Advanced Analytics section"""
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(display_svg("images/ai_analysis.svg", width="200px"), unsafe_allow_html=True)
    st.markdown("<h2>Advanced Analytics</h2>", unsafe_allow_html=True)
    st.markdown("""
    Advanced analytics provides deeper insights into the CA risk factors and prediction model.
    """)
    
    # Check if we have a trained model
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ No trained model available. Please train a model in the System Training section first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Initialize session states for analysis data
    if 'feature_analysis' not in st.session_state:
        st.session_state.feature_analysis = {}
    
    # Advanced analytics tabs
    analytics_tabs = st.tabs([
        "Feature Analysis", 
        "Model Performance", 
        "Risk Distribution by School", 
        "Attendance vs Academic", 
        "Risk Heatmap", 
        "Temporal Trends", 
        "Cohort Analysis",
        "Geographic Mapping",
        "Intervention Cost-Benefit",
        "System Report"
    ])
    
    with analytics_tabs[0]:  # Feature Analysis tab
        st.markdown("<div class='card-title'>🔍 Feature Analysis</div>", unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown("### Key Risk Factors")
        st.markdown("""
        The chart below shows the most influential factors in predicting chronic absenteeism risk, 
        based on the trained model.
        """)
        
        # Feature importance chart
        if 'model' in st.session_state and st.session_state.model is not None:
            fi_fig = plot_feature_importance(st.session_state.model)
            
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
                
                # Get feature importance data for additional analysis
                if hasattr(st.session_state.model, 'feature_importances_'):
                    importances = st.session_state.model.feature_importances_
                    feature_names = getattr(st.session_state.model, 'feature_names_in_', 
                                           [f"Feature {i}" for i in range(len(importances))])
                    
                    # Store for further analysis
                    st.session_state.feature_analysis = dict(zip(feature_names, importances))
            else:
                st.info("Feature importance visualization is not available for this model type.")
        
        # Feature correlation section
        st.markdown("### Feature Correlations")
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            # Compute correlations with target variable if available
            # Check for either CA_Status or CA_Label (for backwards compatibility)
            target_column = None
            if 'CA_Status' in st.session_state.historical_data.columns:
                target_column = 'CA_Status'
            elif 'CA_Label' in st.session_state.historical_data.columns:
                target_column = 'CA_Label'
                
            if target_column is not None:
                # Get numerical columns
                numerical_cols = st.session_state.historical_data.select_dtypes(
                    include=['int64', 'float64']
                ).columns.tolist()
                
                # Remove target from feature list if present
                if target_column in numerical_cols:
                    numerical_cols.remove(target_column)
                
                # Calculate correlations with target
                if numerical_cols:
                    target_correlations = {}
                    
                    for col in numerical_cols:
                        if col in st.session_state.historical_data.columns:
                            correlation = st.session_state.historical_data[col].corr(
                                st.session_state.historical_data[target_column]
                            )
                            if not pd.isna(correlation):
                                target_correlations[col] = correlation
                    
                    # Create DataFrame for plotting
                    corr_df = pd.DataFrame({
                        'Feature': list(target_correlations.keys()),
                        'Correlation': list(target_correlations.values())
                    })
                    
                    # Sort by absolute correlation
                    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
                    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(10)
                    
                    # Plot correlations
                    corr_fig = px.bar(
                        corr_df,
                        x='Correlation',
                        y='Feature',
                        orientation='h',
                        title='Feature Correlations with CA Risk',
                        color='Correlation',
                        color_continuous_scale='RdBu_r',
                        range_color=[-1, 1]
                    )
                    
                    corr_fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=30, b=10),
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(corr_fig, use_container_width=True)
                    
                    # Explain correlations
                    st.markdown("""
                    **Understanding Correlations:**
                    - **Positive values** (blue) indicate that as the feature increases, CA risk tends to increase
                    - **Negative values** (red) indicate that as the feature increases, CA risk tends to decrease
                    - **Values closer to 1 or -1** represent stronger correlations
                    """)
                else:
                    st.info("No numerical features found for correlation analysis.")
            else:
                st.info("Target variable 'CA_Status' not found in data for correlation analysis.")
        else:
            st.info("Historical data not available for correlation analysis.")
        
        # Feature distribution analysis
        st.markdown("### Feature Distributions")
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            # Feature selector
            available_features = st.session_state.historical_data.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            # Filter out certain columns
            exclude_cols = ['Student_ID', 'Date']
            available_features = [f for f in available_features if f not in exclude_cols]
            
            if available_features:
                selected_feature = st.selectbox(
                    "Select Feature to Analyze",
                    options=available_features,
                    index=0 if 'Attendance_Percentage' in available_features else 0
                )
                
                # Check for either CA_Status or CA_Label for target column
                target_column = None
                if 'CA_Status' in st.session_state.historical_data.columns:
                    target_column = 'CA_Status'
                elif 'CA_Label' in st.session_state.historical_data.columns:
                    target_column = 'CA_Label'
                
                # Create distribution based on target if available
                if target_column is not None and selected_feature in st.session_state.historical_data.columns:
                    # Create a histogram grouped by target column
                    hist_fig = px.histogram(
                        st.session_state.historical_data,
                        x=selected_feature,
                        color=target_column,
                        barmode='overlay',
                        opacity=0.7,
                        color_discrete_map={0: 'green', 1: 'red'},
                        labels={target_column: 'Chronic Absenteeism'},
                        title=f'Distribution of {selected_feature} by CA Status'
                    )
                    
                    hist_fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=30, b=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            title=None
                        )
                    )
                    
                    # Update legend labels
                    hist_fig.for_each_trace(lambda t: t.update(name = {0: 'Not CA', 1: 'CA'}[int(t.name)]))
                    
                    st.plotly_chart(hist_fig, use_container_width=True)
                    
                    # Feature statistics
                    st.markdown(f"### {selected_feature} Statistics")
                    
                    # Calculate statistics for the selected feature
                    feature_stats = st.session_state.historical_data.groupby(target_column)[selected_feature].describe()
                    
                    # Rename the index
                    feature_stats.index = ['Not CA', 'CA']
                    
                    # Display statistics
                    st.dataframe(feature_stats, use_container_width=True)
                else:
                    # Create a simple histogram for the selected feature
                    hist_fig = px.histogram(
                        st.session_state.historical_data,
                        x=selected_feature,
                        title=f'Distribution of {selected_feature}'
                    )
                    
                    hist_fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=30, b=10)
                    )
                    
                    st.plotly_chart(hist_fig, use_container_width=True)
                    
                    # Feature statistics
                    st.markdown(f"### {selected_feature} Statistics")
                    
                    # Calculate statistics for the selected feature
                    feature_stats = st.session_state.historical_data[selected_feature].describe()
                    
                    # Display statistics
                    st.dataframe(pd.DataFrame(feature_stats).T, use_container_width=True)
            else:
                st.info("No numerical features found for distribution analysis.")
        else:
            st.info("Historical data not available for distribution analysis.")
    
    with analytics_tabs[1]:  # Model Performance tab
        st.markdown("<div class='card-title'>📊 Model Performance Analysis</div>", unsafe_allow_html=True)
        
        # Check if we have training report
        if 'training_report' in st.session_state and st.session_state.training_report is not None:
            # Model performance metrics
            st.markdown("### Performance Metrics")
            
            # Create a metrics display
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Accuracy", f"{st.session_state.training_report['accuracy']:.4f}")
            with metrics_col2:
                st.metric("Precision", f"{st.session_state.training_report['precision']:.4f}")
            with metrics_col3:
                st.metric("Recall", f"{st.session_state.training_report['recall']:.4f}")
            with metrics_col4:
                st.metric("F1 Score", f"{st.session_state.training_report['f1_score']:.4f}")
            
            # Explain metrics
            with st.expander("Understanding Metrics"):
                st.markdown("""
                **Accuracy**: The proportion of correct predictions (both true positives and true negatives).
                
                **Precision**: The proportion of positive identifications that were actually correct. 
                High precision means few false positives.
                
                **Recall (Sensitivity)**: The proportion of actual positives that were correctly identified.
                High recall means few false negatives.
                
                **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
                """)
            
            # Confusion matrix if available
            if 'confusion_matrix' in st.session_state.training_report:
                st.markdown("### Confusion Matrix Analysis")
                
                cm = st.session_state.training_report['confusion_matrix']
                
                # Calculate derived metrics
                tn, fp, fn, tp = cm.ravel()
                
                total = tn + fp + fn + tp
                accuracy = (tn + tp) / total
                misclassification = (fp + fn) / total
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Display confusion matrix
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Not CA', 'Predicted CA'],
                    y=['Actual Not CA', 'Actual CA'],
                    colorscale='Blues',
                    showscale=False,
                    text=[[tn, fp], [fn, tp]],
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ))
                
                cm_fig.update_layout(
                    title='Confusion Matrix',
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=10)
                )
                
                st.plotly_chart(cm_fig, use_container_width=True)
                
                # Create derived metrics display
                derived_col1, derived_col2, derived_col3, derived_col4 = st.columns(4)
                
                with derived_col1:
                    st.metric("True Positives", tp)
                    st.metric("False Positives", fp)
                with derived_col2:
                    st.metric("True Negatives", tn)
                    st.metric("False Negatives", fn)
                with derived_col3:
                    st.metric("Specificity", f"{specificity:.4f}")
                    st.metric("Misclassification", f"{misclassification:.4f}")
                with derived_col4:
                    st.metric("Precision", f"{precision:.4f}")
                    st.metric("Recall", f"{recall:.4f}")
                
                # Explain confusion matrix
                st.markdown("""
                **Understanding the Confusion Matrix:**
                - **True Positives (TP)**: Students correctly predicted as CA
                - **False Positives (FP)**: Students incorrectly predicted as CA
                - **True Negatives (TN)**: Students correctly predicted as not CA
                - **False Negatives (FN)**: Students incorrectly predicted as not CA
                
                **Specificity**: The proportion of actual negatives correctly identified (TN / (TN + FP))
                
                **Misclassification Rate**: The proportion of incorrect predictions ((FP + FN) / Total)
                """)
            
            # ROC curve if available
            if 'y_pred_proba' in st.session_state.training_report and st.session_state.training_report['y_pred_proba'] is not None:
                st.markdown("### Receiver Operating Characteristic (ROC) Curve")
                
                # This would require the test labels and prediction probabilities
                # Since we don't have them stored, we'll show a placeholder message
                st.info("ROC curve visualization requires test set data to be retained. This feature will be available in a future update.")
        else:
            st.info("Training report not available. Please train a model first.")
    
    # Risk Distribution by School tab
    with analytics_tabs[2]:
        st.markdown("<div class='card-title'>🏫 Risk Distribution by School</div>", unsafe_allow_html=True)
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            if 'School' in st.session_state.historical_data.columns:
                # Check for risk column
                risk_column = None
                if 'CA_Risk' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Risk'
                elif 'CA_Status' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Status'
                
                if risk_column:
                    # Group by school and calculate risk statistics
                    school_risk = st.session_state.historical_data.groupby('School')[risk_column].agg(['mean', 'count'])
                    school_risk = school_risk.reset_index()
                    school_risk.columns = ['School', 'Average_Risk', 'Student_Count']
                    
                    # Create visualization
                    st.markdown("### Average Risk Score by School")
                    
                    # Bar chart
                    fig1 = px.bar(
                        school_risk.sort_values('Average_Risk', ascending=False),
                        x='School',
                        y='Average_Risk',
                        color='Average_Risk',
                        color_continuous_scale='RdYlGn_r',
                        labels={'Average_Risk': 'Average CA Risk'},
                        hover_data=['Student_Count'],
                        height=400
                    )
                    
                    fig1.update_layout(
                        xaxis_title="School",
                        yaxis_title="Average Risk Score",
                        coloraxis_showscale=True,
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Risk distribution
                    st.markdown("### Risk Category Distribution by School")
                    
                    # Calculate risk categories if not present
                    if 'Risk_Category' not in st.session_state.historical_data.columns:
                        df = st.session_state.historical_data.copy()
                        df['Risk_Category'] = df[risk_column].apply(lambda x: 
                            'High' if x >= 0.7 else ('Medium' if x >= 0.3 else 'Low'))
                    else:
                        df = st.session_state.historical_data
                    
                    # Group by school and risk category
                    school_cat_counts = pd.crosstab(
                        df['School'], 
                        df['Risk_Category'],
                        normalize='index'
                    ) * 100
                    
                    # Plot stacked bar chart
                    fig2 = px.bar(
                        school_cat_counts.reset_index(),
                        x='School',
                        y=['High', 'Medium', 'Low'] if all(c in school_cat_counts.columns for c in ['High', 'Medium', 'Low']) 
                          else school_cat_counts.columns.tolist(),
                        title="Risk Category Distribution by School (%)",
                        labels={'value': 'Percentage', 'variable': 'Risk Category'},
                        color_discrete_map={'High': '#FF9999', 'Medium': '#FFCC99', 'Low': '#99CC99'}
                    )
                    
                    fig2.update_layout(
                        xaxis_title="School",
                        yaxis_title="Percentage",
                        legend_title="Risk Category",
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Risk score column not found in historical data.")
            else:
                st.info("School information not found in historical data.")
        else:
            st.info("Historical data not available for analysis.")
    
    # Attendance vs. Academic Performance tab
    with analytics_tabs[3]:
        st.markdown("<div class='card-title'>📚 Attendance vs Academic Performance</div>", unsafe_allow_html=True)
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            # Check for required columns
            attendance_col = None
            academic_col = None
            
            # Find attendance column
            for col in ['Attendance_Percentage', 'Present_Days', 'Absent_Days']:
                if col in st.session_state.historical_data.columns:
                    attendance_col = col
                    break
            
            # Find academic performance column
            if 'Academic_Performance' in st.session_state.historical_data.columns:
                academic_col = 'Academic_Performance'
            
            if attendance_col and academic_col:
                # Scatter plot with risk coloring
                st.markdown("### Attendance vs. Academic Performance Correlation")
                
                # Check for risk column for coloring
                risk_column = None
                if 'CA_Risk' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Risk'
                elif 'CA_Status' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Status'
                
                # Create scatter plot
                if risk_column:
                    fig = px.scatter(
                        st.session_state.historical_data,
                        x=attendance_col,
                        y=academic_col,
                        color=risk_column,
                        color_continuous_scale='RdYlGn_r',
                        opacity=0.7,
                        title=f"Attendance vs. Academic Performance",
                        labels={
                            attendance_col: "Attendance",
                            academic_col: "Academic Performance",
                            risk_column: "CA Risk"
                        },
                        height=500
                    )
                else:
                    fig = px.scatter(
                        st.session_state.historical_data,
                        x=attendance_col,
                        y=academic_col,
                        opacity=0.7,
                        title=f"Attendance vs. Academic Performance",
                        labels={
                            attendance_col: "Attendance",
                            academic_col: "Academic Performance"
                        },
                        height=500
                    )
                
                fig.update_layout(
                    xaxis_title="Attendance",
                    yaxis_title="Academic Performance",
                    margin=dict(l=20, r=20, t=30, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation
                correlation = st.session_state.historical_data[attendance_col].corr(
                    st.session_state.historical_data[academic_col]
                )
                
                st.markdown(f"**Correlation Coefficient**: {correlation:.4f}")
                
                # Interpretation
                st.markdown("""
                **Interpretation:**
                - A correlation closer to +1 indicates a strong positive relationship between attendance and academic performance
                - Values closer to 0 indicate weaker or no linear relationship
                - Negative values would suggest that as attendance increases, academic performance decreases (which would be unusual)
                """)
                
                # Linear regression model
                st.markdown("### Predictive Relationship")
                st.markdown("Fitting a simple linear model to predict academic performance from attendance:")
                
                # Simple linear regression
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                X = st.session_state.historical_data[attendance_col].values.reshape(-1, 1)
                y = st.session_state.historical_data[academic_col].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Create prediction line
                # Create numerical range for prediction line using numpy
                import numpy as np
                x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_pred = model.predict(x_range)
                
                # Plot with regression line
                fig2 = px.scatter(
                    st.session_state.historical_data,
                    x=attendance_col,
                    y=academic_col,
                    opacity=0.7,
                    title="Attendance vs. Academic Performance with Trend Line",
                    labels={
                        attendance_col: "Attendance",
                        academic_col: "Academic Performance"
                    },
                    height=500
                )
                
                # Add regression line
                fig2.add_traces(
                    go.Scatter(
                        x=x_range.reshape(-1),
                        y=y_pred,
                        mode='lines',
                        name='Predicted Trend',
                        line=dict(color='red', width=2)
                    )
                )
                
                fig2.update_layout(
                    xaxis_title="Attendance",
                    yaxis_title="Academic Performance",
                    margin=dict(l=20, r=20, t=30, b=10),
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Display regression statistics
                y_pred_all = model.predict(X)
                r2 = r2_score(y, y_pred_all)
                
                st.markdown(f"""
                **Regression Analysis:**
                - Slope (coefficient): {model.coef_[0]:.4f}
                - Intercept: {model.intercept_:.4f}
                - R² (coefficient of determination): {r2:.4f}
                
                **Interpretation:** For each 1% increase in attendance, academic performance changes by {model.coef_[0]:.4f} points.
                The R² value indicates that {r2*100:.1f}% of the variation in academic performance can be explained by attendance.
                """)
            else:
                st.info("Required columns for this analysis are not available in the historical data.")
        else:
            st.info("Historical data not available for analysis.")
    
    # Risk Heatmap by Grade & SES tab
    with analytics_tabs[4]:
        st.markdown("<div class='card-title'>🔥 Risk Heatmap by Grade & SES</div>", unsafe_allow_html=True)
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            # Check for required columns
            if 'Grade' in st.session_state.historical_data.columns and 'Meal_Code' in st.session_state.historical_data.columns:
                # Check for risk column
                risk_column = None
                if 'CA_Risk' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Risk'
                elif 'CA_Status' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Status'
                
                if risk_column:
                    # Create pivot table
                    pivot_data = st.session_state.historical_data.pivot_table(
                        values=risk_column,
                        index='Grade',
                        columns='Meal_Code',
                        aggfunc='mean'
                    ).round(2)
                    
                    # Ensure all meal codes are included
                    for meal_code in ['Free', 'Reduced', 'Paid']:
                        if meal_code not in pivot_data.columns:
                            pivot_data[meal_code] = pd.NA
                    
                    # Sort grade levels numerically
                    if pivot_data.index.dtype == 'int64' or pivot_data.index.dtype == 'float64':
                        pivot_data = pivot_data.sort_index()
                    
                    # Create heatmap
                    st.markdown("### CA Risk by Grade and Socioeconomic Status")
                    
                    # Convert to format suitable for heatmap
                    heatmap_data = pivot_data.reset_index().melt(
                        id_vars='Grade',
                        value_vars=pivot_data.columns,
                        var_name='Meal_Code',
                        value_name='CA_Risk'
                    )
                    
                    # Plotly heatmap
                    fig = px.imshow(
                        pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        color_continuous_scale='RdYlGn_r',
                        labels=dict(x="Meal Code (SES)", y="Grade", color="CA Risk"),
                        title="CA Risk Heatmap by Grade and Socioeconomic Status"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Socioeconomic Status (Meal Code)",
                        yaxis_title="Grade",
                        coloraxis_colorbar=dict(title="CA Risk"),
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    
                    # Add text annotations
                    fig.update_traces(text=pivot_data.values, texttemplate="%{text}", textfont={"size": 10})
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - Darker red colors indicate higher risk of chronic absenteeism
                    - Darker green colors indicate lower risk of chronic absenteeism
                    - This heatmap helps identify specific grade and socioeconomic combinations that may require targeted interventions
                    
                    **SES (Socioeconomic Status) Legend:**
                    - Free: Students eligible for free meals (lower SES)
                    - Reduced: Students eligible for reduced-price meals (middle SES)
                    - Paid: Students paying full price for meals (higher SES)
                    """)
                    
                    # Show the underlying data
                    st.markdown("### Risk by Grade and SES (Data Table)")
                    st.dataframe(pivot_data, use_container_width=True)
                else:
                    st.info("Risk score column not found in historical data.")
            else:
                st.info("Required columns (Grade, Meal_Code) not found in historical data.")
        else:
            st.info("Historical data not available for analysis.")

    # Temporal Attendance Trends tab
    with analytics_tabs[5]:
        st.markdown("<div class='card-title'>📅 Temporal Attendance Trends</div>", unsafe_allow_html=True)
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            # Check for 'Year' column
            if 'Year' in st.session_state.historical_data.columns:
                # Check for attendance columns
                attendance_col = None
                for col in ['Attendance_Percentage', 'Present_Days', 'Absent_Days']:
                    if col in st.session_state.historical_data.columns:
                        attendance_col = col
                        break
                
                if attendance_col:
                    # Group by year
                    yearly_trends = st.session_state.historical_data.groupby('Year')[attendance_col].mean().reset_index()
                    
                    # Create time series visualization
                    st.markdown("### Attendance Trends Over Time")
                    
                    fig = px.line(
                        yearly_trends,
                        x='Year',
                        y=attendance_col,
                        markers=True,
                        labels={attendance_col: "Average Attendance"},
                        title="Attendance Trends by Year"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Average Attendance",
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add year-on-year change calculation
                    if len(yearly_trends) > 1:
                        yearly_trends['Change'] = yearly_trends[attendance_col].diff()
                        yearly_trends['Percent_Change'] = yearly_trends[attendance_col].pct_change() * 100
                        
                        # Display change data
                        st.markdown("### Year-on-Year Attendance Changes")
                        
                        # Format the change data
                        display_df = yearly_trends.copy()
                        display_df[attendance_col] = display_df[attendance_col].round(2)
                        display_df['Change'] = display_df['Change'].round(2)
                        display_df['Percent_Change'] = display_df['Percent_Change'].round(2)
                        
                        # Add % symbol to percent change
                        display_df['Percent_Change'] = display_df['Percent_Change'].apply(
                            lambda x: f"{x}%" if not pd.isna(x) else ""
                        )
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Create a bar chart for changes
                        fig2 = px.bar(
                            yearly_trends.iloc[1:],  # Skip first row (no change data)
                            x='Year',
                            y='Change',
                            labels={'Change': f"Change in {attendance_col}"},
                            title="Year-on-Year Attendance Change",
                            color='Change',
                            color_continuous_scale='RdYlGn'
                        )
                        
                        fig2.update_layout(
                            xaxis_title="Year",
                            yaxis_title=f"Change in {attendance_col}",
                            margin=dict(l=20, r=20, t=30, b=10),
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # School-level trends if school data is available
                    if 'School' in st.session_state.historical_data.columns:
                        st.markdown("### Attendance Trends by School")
                        
                        # Group by year and school
                        school_trends = st.session_state.historical_data.groupby(['Year', 'School'])[attendance_col].mean().reset_index()
                        
                        # Create multi-line chart
                        fig3 = px.line(
                            school_trends,
                            x='Year',
                            y=attendance_col,
                            color='School',
                            markers=True,
                            labels={attendance_col: "Average Attendance"},
                            title="Attendance Trends by School and Year"
                        )
                        
                        fig3.update_layout(
                            xaxis_title="Year",
                            yaxis_title="Average Attendance",
                            legend_title="School",
                            margin=dict(l=20, r=20, t=30, b=10),
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Attendance data columns not found in historical data.")
            else:
                st.info("Year information not found in historical data.")
        else:
            st.info("Historical data not available for analysis.")
    
    # Cohort Analysis tab
    with analytics_tabs[6]:
        st.markdown("<div class='card-title'>👨‍👩‍👧‍👦 Cohort Analysis</div>", unsafe_allow_html=True)
        
        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
            # Check for demographic columns
            demographic_cols = []
            for col in ['Gender', 'Meal_Code', 'Grade']:
                if col in st.session_state.historical_data.columns:
                    demographic_cols.append(col)
            
            if demographic_cols:
                # Let user select demographic for cohort analysis
                selected_demographic = st.selectbox(
                    "Select Demographic for Cohort Analysis",
                    options=demographic_cols,
                    index=0
                )
                
                # Check for risk column
                risk_column = None
                if 'CA_Risk' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Risk'
                elif 'CA_Status' in st.session_state.historical_data.columns:
                    risk_column = 'CA_Status'
                
                if risk_column:
                    # Group by selected demographic
                    cohort_risk = st.session_state.historical_data.groupby(selected_demographic)[risk_column].agg(['mean', 'count'])
                    cohort_risk = cohort_risk.reset_index()
                    cohort_risk.columns = [selected_demographic, 'Average_Risk', 'Student_Count']
                    
                    # Sort by risk
                    cohort_risk = cohort_risk.sort_values('Average_Risk', ascending=False)
                    
                    # Create visualization
                    st.markdown(f"### CA Risk by {selected_demographic}")
                    
                    fig = px.bar(
                        cohort_risk,
                        x=selected_demographic,
                        y='Average_Risk',
                        color='Average_Risk',
                        color_continuous_scale='RdYlGn_r',
                        labels={'Average_Risk': 'Average CA Risk'},
                        hover_data=['Student_Count'],
                        height=400,
                        title=f"Average CA Risk by {selected_demographic}"
                    )
                    
                    fig.update_layout(
                        xaxis_title=selected_demographic,
                        yaxis_title="Average Risk Score",
                        coloraxis_showscale=True,
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add student count visualization
                    st.markdown(f"### Student Count by {selected_demographic}")
                    
                    fig2 = px.pie(
                        cohort_risk,
                        names=selected_demographic,
                        values='Student_Count',
                        title=f"Student Distribution by {selected_demographic}",
                        hover_data=['Average_Risk'],
                        height=400
                    )
                    
                    fig2.update_traces(textposition='inside', textinfo='percent+label')
                    fig2.update_layout(margin=dict(l=20, r=20, t=30, b=10))
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Show data table
                    st.markdown(f"### {selected_demographic} Cohort Data")
                    st.dataframe(cohort_risk, use_container_width=True)
                    
                    # Allow for secondary demographic split
                    if len(demographic_cols) > 1:
                        st.markdown("### Multi-Dimensional Cohort Analysis")
                        
                        # Filter out the already selected demographic
                        secondary_options = [col for col in demographic_cols if col != selected_demographic]
                        
                        secondary_demographic = st.selectbox(
                            "Select Secondary Demographic",
                            options=secondary_options,
                            index=0
                        )
                        
                        # Create grouped analysis
                        grouped_data = st.session_state.historical_data.pivot_table(
                            values=risk_column,
                            index=selected_demographic,
                            columns=secondary_demographic,
                            aggfunc='mean'
                        ).round(2)
                        
                        # Heatmap for the grouped data
                        st.markdown(f"### CA Risk by {selected_demographic} and {secondary_demographic}")
                        
                        fig3 = px.imshow(
                            grouped_data.values,
                            x=grouped_data.columns,
                            y=grouped_data.index,
                            color_continuous_scale='RdYlGn_r',
                            labels=dict(x=secondary_demographic, y=selected_demographic, color="CA Risk"),
                            title=f"CA Risk by {selected_demographic} and {secondary_demographic}"
                        )
                        
                        fig3.update_layout(
                            xaxis_title=secondary_demographic,
                            yaxis_title=selected_demographic,
                            coloraxis_colorbar=dict(title="CA Risk"),
                            margin=dict(l=20, r=20, t=30, b=10),
                        )
                        
                        # Add text annotations
                        fig3.update_traces(text=grouped_data.values, texttemplate="%{text}", textfont={"size": 10})
                        
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Show data table
                        st.markdown(f"### Detailed Risk by {selected_demographic} and {secondary_demographic}")
                        st.dataframe(grouped_data, use_container_width=True)
                else:
                    st.info("Risk score column not found in historical data.")
            else:
                st.info("No demographic columns found for cohort analysis.")
        else:
            st.info("Historical data not available for analysis.")
    
    # Geographic Risk Mapping tab
    with analytics_tabs[7]:
        st.markdown("<div class='card-title'>🗺️ Geographic Risk Mapping</div>", unsafe_allow_html=True)
        
        # Since we don't have actual geographic data in our dataset, we'll create a simulated district map
        st.markdown("""
        ### Geographic Risk Distribution
        
        This visualization would typically show a geographic heatmap of CA risk across different school zones or neighborhoods.
        
        **Example uses:**
        - Identify geographic clusters of high CA risk
        - Target resources to high-need areas
        - Analyze transportation or community factors affecting attendance
        
        **Required data:**
        - School geographic coordinates or zone boundaries
        - Student address data (anonymized)
        - Risk scores mapped to geographic areas
        
        To implement actual geographic mapping, geographic data would need to be collected and added to the system.
        """)
        
        # Create a sample image to demonstrate concept
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a placeholder map
        st.markdown("### Sample District Map (Simulated Data)")
        
        # Create a sample map using plotly
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        
        # Create some simulated risk data with spatial clustering
        np.random.seed(42)  # For reproducibility
        Z = np.zeros((20, 20))
        
        # Add some "hot spots" of risk
        Z[3:8, 3:8] = 0.8  # High risk cluster
        Z[12:17, 12:17] = 0.7  # Another high risk area
        Z[12:17, 2:6] = 0.4  # Medium risk area
        
        # Add some random noise
        Z += np.random.normal(0, 0.1, Z.shape)
        Z = np.clip(Z, 0, 1)  # Ensure values are between 0 and 1
        
        # Create a heatmap
        fig = go.Figure(data=go.Heatmap(
            z=Z,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="CA Risk")
        ))
        
        # Add school markers (simulated locations)
        schools = [
            {"name": "North High", "x": 5, "y": 5, "risk": 0.65},
            {"name": "South High", "x": 15, "y": 15, "risk": 0.78},
            {"name": "East Middle", "x": 15, "y": 4, "risk": 0.45},
            {"name": "West Elementary", "x": 4, "y": 15, "risk": 0.32},
            {"name": "Central Academy", "x": 10, "y": 10, "risk": 0.55}
        ]
        
        for school in schools:
            fig.add_trace(go.Scatter(
                x=[school["x"]],
                y=[school["y"]],
                mode="markers+text",
                marker=dict(
                    size=15,
                    color='black',
                    symbol='star'
                ),
                text=[school["name"]],
                textposition="top center",
                name=school["name"],
                hoverinfo="text",
                hovertext=f"{school['name']}: {school['risk']:.2f} risk"
            ))
        
        fig.update_layout(
            title="District CA Risk Map (Simulated)",
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note:** This is a simulated visualization. To implement actual geographic mapping, 
        you would need to collect geographic data for schools and student residences, 
        then use a library like Plotly or Folium to create interactive maps.
        """)
    
    # Intervention Cost-Benefit tab
    with analytics_tabs[8]:
        st.markdown("<div class='card-title'>💰 Intervention Cost-Benefit Analysis</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ### Intervention ROI Calculator
        
        This tool helps estimate the return on investment (ROI) for different intervention strategies.
        
        **Directions:** Adjust the parameters below to model the impact and cost-effectiveness of various intervention approaches.
        """)
        
        # Parameters for the calculator
        cost_col1, cost_col2 = st.columns(2)
        
        with cost_col1:
            total_students = st.number_input("Total Student Population", min_value=10, max_value=10000, value=1000, step=10)
            ca_percentage = st.number_input("Current CA Percentage", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
            per_student_funding = st.number_input("Annual Per-Student Funding ($)", min_value=1000, max_value=50000, value=12000, step=500)
        
        with cost_col2:
            intervention_cost = st.number_input("Intervention Cost Per Student ($)", min_value=0, max_value=5000, value=500, step=50)
            estimated_improvement = st.slider("Estimated Percentage Point Reduction in CA", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
            attendance_funding_weight = st.slider("Attendance Funding Weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05, 
                                               help="Proportion of funding tied to attendance")
        
        # Calculate metrics
        ca_students = total_students * (ca_percentage / 100)
        intervention_total_cost = intervention_cost * ca_students
        
        # Improved scenario
        improved_ca_percentage = max(0, ca_percentage - estimated_improvement)
        improved_ca_students = total_students * (improved_ca_percentage / 100)
        students_helped = ca_students - improved_ca_students
        
        # Financial impact
        current_funding_loss = total_students * per_student_funding * attendance_funding_weight * (ca_percentage / 100)
        improved_funding_loss = total_students * per_student_funding * attendance_funding_weight * (improved_ca_percentage / 100)
        funding_gain = current_funding_loss - improved_funding_loss
        
        # ROI metrics
        net_benefit = funding_gain - intervention_total_cost
        roi_percent = (net_benefit / intervention_total_cost * 100) if intervention_total_cost > 0 else 0
        
        # Display results
        st.markdown("### Cost-Benefit Analysis Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Current CA Students", f"{int(ca_students)}")
            st.metric("Projected CA Students After Intervention", f"{int(improved_ca_students)}")
            st.metric("Students Helped", f"{int(students_helped)}")
        
        with result_col2:
            st.metric("Total Intervention Cost", f"${intervention_total_cost:,.2f}")
            st.metric("Current Annual Funding Loss", f"${current_funding_loss:,.2f}")
            st.metric("Projected Annual Funding Loss", f"${improved_funding_loss:,.2f}")
        
        with result_col3:
            st.metric("Annual Funding Recaptured", f"${funding_gain:,.2f}")
            st.metric("Net Annual Benefit", f"${net_benefit:,.2f}")
            st.metric("ROI", f"{roi_percent:.1f}%")
        
        # ROI visualization
        st.markdown("### Intervention ROI Visualization")
        
        # Create bar chart for ROI components
        roi_data = pd.DataFrame([
            {"Category": "Intervention Cost", "Amount": -intervention_total_cost},
            {"Category": "Funding Recaptured", "Amount": funding_gain},
            {"Category": "Net Benefit", "Amount": net_benefit}
        ])
        
        # Determine color based on value
        roi_data["Color"] = roi_data["Amount"].apply(
            lambda x: "red" if x < 0 else "green"
        )
        
        fig = px.bar(
            roi_data,
            x="Category",
            y="Amount",
            title="Intervention ROI Breakdown",
            color="Color",
            color_discrete_map={"red": "red", "green": "green"},
            height=400
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Amount ($)",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        
        # Add dollar signs to y-axis labels
        fig.update_yaxes(tickprefix="$")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - Positive ROI indicates that the intervention is cost-effective
        - Higher ROI suggests better financial return on the investment
        - This analysis only considers direct funding impacts; other benefits like improved student outcomes and reduced social costs are not included
        
        **Note:** This is a simplified model. Actual results will depend on many factors including intervention effectiveness, 
        implementation quality, and how funding formulas are structured in your district.
        """)
    
    # System Report tab (moved to the end)
    with analytics_tabs[9]:  # System Report tab
        st.markdown("<div class='card-title'>📑 System Report</div>", unsafe_allow_html=True)
        
        # System summary
        st.markdown("### System Summary")
        
        # Check if model is available
        if 'model' in st.session_state and st.session_state.model is not None:
            # Display system information
            system_col1, system_col2 = st.columns(2)
            
            with system_col1:
                st.metric("Model Type", st.session_state.active_model.replace('_', ' ').title() if 'active_model' in st.session_state else 'Unknown')
                
                if 'historical_data' in st.session_state:
                    st.metric("Training Data Size", len(st.session_state.historical_data))
            
            with system_col2:
                if 'training_report' in st.session_state:
                    st.metric("Model Accuracy", f"{st.session_state.training_report['accuracy']:.4f}")
                    st.metric("F1 Score", f"{st.session_state.training_report['f1_score']:.4f}")
            
            # View saved reports
            st.markdown("### Saved Reports")
            
            # Check for saved reports in session state
            if 'saved_reports' in st.session_state and st.session_state.saved_reports:
                # Display previously generated reports
                for i, report in enumerate(reversed(st.session_state.saved_reports)):
                    try:
                        # Format the timestamp for display
                        timestamp = datetime.fromisoformat(report['timestamp'])
                        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Create an expander for each report
                        with st.expander(f"Report {time_str} - {report['model_type'].replace('_', ' ').title()}", expanded=(i==0)):
                            # Show a preview of the report content
                            preview_lines = report['content'].split('\n')[:15]
                            st.code('\n'.join(preview_lines) + '\n...', language='text')
                            
                            # Create download button for the report
                            st.download_button(
                                "Download Full Report",
                                report['content'],
                                report['filename'],
                                "text/plain",
                                key=f"download_report_{i}"
                            )
                    except Exception as e:
                        st.error(f"Error displaying report: {str(e)}")
            else:
                st.info("No saved reports found. Generate a new report below.")
            
            # Generate a new report
            st.markdown("### Generate New Report")
            st.markdown("""
            Generate a comprehensive system performance report with detailed metrics and analysis.
            This report can be downloaded as a text file for documentation and review.
            """)
            
            # Report generation button
            if st.button("Generate System Report", key="generate_report_button"):
                generate_system_report()
        else:
            st.info("No trained model available. Please train a model first to generate a system report.")
    
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