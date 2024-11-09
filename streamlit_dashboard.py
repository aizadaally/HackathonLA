import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def load_data():
    """Load all necessary data files."""
    df = pd.read_csv('incident_data.csv')
    frequency_metrics = pd.read_csv('frequency_analysis.csv')
    predictions = pd.read_csv('incident_predictions.csv')

    # Convert date columns
    df['Incident_Date'] = pd.to_datetime(df['Incident_Date']).dt.date
    df['DOB_of_Abuser'] = pd.to_datetime(df['DOB_of_Abuser']).dt.date
    predictions['next_incident_date'] = pd.to_datetime(predictions['next_incident_date']).dt.date

    return df, frequency_metrics, predictions

def create_risk_dashboard():
    """Create the main dashboard."""
    st.title("Domestic Violence Risk Assessment Dashboard")

    # Load data
    df, frequency_metrics, predictions = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_oblast = st.sidebar.selectbox(
        "Select Region (Oblast)",
        options=['All'] + sorted(df['Oblast'].unique().tolist())
    )

    min_severity = st.sidebar.slider(
        "Minimum Severity Score",
        min_value=1,
        max_value=10,
        value=6
    )

    # Filter data
    if selected_oblast != 'All':
        df = df[df['Oblast'] == selected_oblast]

    # Combine all metrics
    dashboard_data = pd.merge(df, frequency_metrics, on='Abuser_ID', how='left')
    dashboard_data = pd.merge(dashboard_data, predictions, on='Abuser_ID', how='left')

    # Filter by severity
    high_risk_cases = dashboard_data[dashboard_data['Severity_of_Case'] >= min_severity]

    # Main dashboard content
    st.header(f"High-Risk Cases - {selected_oblast}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total High-Risk Cases", len(high_risk_cases['Abuser_ID'].unique()))
    with col2:
        st.metric("Average Severity", f"{high_risk_cases['Severity_of_Case'].mean():.1f}")
    with col3:
        st.metric("Hospitalization Rate", 
                 f"{high_risk_cases['hospitalization_rate'].mean():.1f}%")

    # High-risk cases table
    st.subheader("High-Risk Abusers")
    high_risk_table = high_risk_cases.groupby('Abuser_ID').agg({
        'Gender_of_Abuser': 'first',
        'DOB_of_Abuser': 'first',
        'next_incident_date': 'first',
        'Severity_of_Case': 'mean'
    }).reset_index()

    high_risk_table = high_risk_table.sort_values('Severity_of_Case', ascending=False)

    # Rename columns and display the DataFrame
    st.dataframe(
        high_risk_table.rename(columns={
            'Abuser_ID': 'ID',
            'Gender_of_Abuser': 'Gender',
            'DOB_of_Abuser': 'Date of Birth',
            'next_incident_date': 'Predicted Next Incident',
            'Severity_of_Case': 'Average Severity'
        }),
        use_container_width=True
    )

    # Severity distribution plot
    st.subheader("Severity Distribution by Oblast")
    fig = px.box(dashboard_data, x='Oblast', y='Severity_of_Case',
                 title='Distribution of Case Severity by Region')
    st.plotly_chart(fig)

if __name__ == "__main__":
    st.set_page_config(page_title="DV Risk Assessment", layout="wide")
    create_risk_dashboard()