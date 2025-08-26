import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

def display_user_history(reports: List[Dict[str, Any]], firebase_auth):
    """
    Display user's analysis history in a formatted way.
    
    Args:
        reports (List[Dict]): List of user's reports
        firebase_auth: Firebase authentication instance
    """
    if not reports:
        st.info("📋 No previous reports found. Start your first analysis!")
        return
    
    st.markdown("### 📊 Analysis History")
    
    # Create a summary section
    col1, col2, col3, col4 = st.columns(4)
    
    total_analyses = len(reports)
    tumor_types = [report.get('prediction', {}).get('tumor_type', 'unknown') for report in reports]
    malignant_count = sum(1 for report in reports if report.get('prediction', {}).get('malignancy') == 'malignant')
    benign_count = total_analyses - malignant_count
    
    with col1:
        st.metric("Total Analyses", total_analyses)
    with col2:
        st.metric("Malignant Cases", malignant_count)
    with col3:
        st.metric("Benign Cases", benign_count)
    with col4:
        avg_confidence = sum(report.get('prediction', {}).get('confidence', 0) for report in reports) / total_analyses
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Create detailed reports table
    st.markdown("### 📋 Recent Reports")
    
    # Prepare data for the table
    table_data = []
    for i, report in enumerate(reports[:10]):  # Show last 10 reports
        prediction = report.get('prediction', {})
        created_at = report.get('created_at', datetime.now())
        
        if isinstance(created_at, datetime):
            date_str = created_at.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(created_at)
        
        table_data.append({
            "Date": date_str,
            "Tumor Type": prediction.get('tumor_type', 'Unknown').replace('_', ' ').title(),
            "Malignancy": prediction.get('malignancy', 'Unknown').title(),
            "Confidence": f"{prediction.get('confidence', 0):.1%}",
            "Report ID": report.get('id', f"Report {i+1}")
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    # Create visualizations
    st.markdown("### 📈 Analysis Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tumor type distribution
        tumor_counts = pd.Series(tumor_types).value_counts()
        fig1 = px.pie(
            values=tumor_counts.values,
            names=tumor_counts.index,
            title="Tumor Type Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Malignancy distribution
        malignancy_data = [report.get('prediction', {}).get('malignancy', 'unknown') for report in reports]
        malignancy_counts = pd.Series(malignancy_data).value_counts()
        fig2 = px.bar(
            x=malignancy_counts.index,
            y=malignancy_counts.values,
            title="Malignancy Distribution",
            color=malignancy_counts.index,
            color_discrete_map={'malignant': '#dc3545', 'benign': '#28a745'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Confidence trend over time
    st.markdown("### 📊 Confidence Trend")
    dates = []
    confidences = []
    
    for report in reports:
        prediction = report.get('prediction', {})
        created_at = report.get('created_at', datetime.now())
        
        if isinstance(created_at, datetime):
            dates.append(created_at)
        else:
            dates.append(datetime.now())
        
        confidences.append(prediction.get('confidence', 0))
    
    if dates and confidences:
        trend_df = pd.DataFrame({
            'Date': dates,
            'Confidence': confidences
        })
        
        fig3 = px.line(
            trend_df,
            x='Date',
            y='Confidence',
            title="Confidence Trend Over Time",
            markers=True
        )
        fig3.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig3, use_container_width=True)

def display_report_details(report: Dict[str, Any]):
    """
    Display detailed information for a specific report.
    
    Args:
        report (Dict): Report data to display
    """
    prediction = report.get('prediction', {})
    medical_summary = report.get('medical_summary', '')
    
    st.markdown("### 📋 Report Details")
    
    # Key findings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Tumor Type", 
            prediction.get('tumor_type', 'Unknown').replace('_', ' ').title()
        )
    
    with col2:
        malignancy = prediction.get('malignancy', 'Unknown').title()
        color = "#dc3545" if malignancy == "Malignant" else "#28a745"
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Malignancy</h4>
            <p style="color: {color}; font-weight: bold; font-size: 1.2em;">{malignancy}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric(
            "Confidence", 
            f"{prediction.get('confidence', 0):.1%}"
        )
    
    # Medical summary
    st.markdown("### 📄 Medical Summary")
    st.text_area(
        "Analysis Report",
        value=medical_summary,
        height=300,
        disabled=True
    )
    
    # Date information
    created_at = report.get('created_at', datetime.now())
    if isinstance(created_at, datetime):
        st.info(f"📅 Analysis performed on: {created_at.strftime('%Y-%m-%d at %H:%M:%S')}")

def create_report_summary_card(report: Dict[str, Any], index: int):
    """
    Create a summary card for a report in the history view.
    
    Args:
        report (Dict): Report data
        index (int): Report index
        
    Returns:
        bool: True if user wants to view details
    """
    prediction = report.get('prediction', {})
    created_at = report.get('created_at', datetime.now())
    
    if isinstance(created_at, datetime):
        date_str = created_at.strftime("%Y-%m-%d %H:%M")
    else:
        date_str = str(created_at)
    
    tumor_type = prediction.get('tumor_type', 'Unknown').replace('_', ' ').title()
    malignancy = prediction.get('malignancy', 'Unknown').title()
    confidence = prediction.get('confidence', 0)
    
    # Color coding for malignancy
    malignancy_color = "#dc3545" if malignancy == "Malignant" else "#28a745"
    
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0;">{tumor_type}</h4>
                    <p style="margin: 5px 0; color: {malignancy_color}; font-weight: bold;">
                        {malignancy}
                    </p>
                    <p style="margin: 5px 0; color: #666;">
                        Confidence: {confidence:.1%} | Date: {date_str}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(f"📋 View Details", key=f"view_{index}"):
                return True
        with col2:
            if st.button(f"🗑️ Delete", key=f"delete_{index}"):
                return "delete"
    
    return False
