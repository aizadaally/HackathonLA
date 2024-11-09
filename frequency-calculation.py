import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data(file_path):
    """Load and prepare the incident data for analysis."""
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime
    df['Incident_Date'] = pd.to_datetime(df['Incident_Date'])
    df['DOB_of_Abuser'] = pd.to_datetime(df['DOB_of_Abuser'])
    
    return df

def calculate_incident_frequency(df):
    """Calculate frequency metrics for each abuser."""
    # Count incidents per abuser
    incident_counts = df.groupby('Abuser_ID').size().reset_index(name='incident_count')
    
    # Calculate time gaps between incidents for each abuser
    time_gaps = df.sort_values('Incident_Date').groupby('Abuser_ID').agg({
        'Incident_Date': lambda x: x.diff().dt.days.mean()
    }).reset_index()
    time_gaps.columns = ['Abuser_ID', 'avg_days_between_incidents']
    
    # Merge frequency metrics
    frequency_metrics = pd.merge(incident_counts, time_gaps, on='Abuser_ID', how='left')
    
    # Fill NaN values for single-incident abusers with a default value (e.g., 365 days)
    frequency_metrics['avg_days_between_incidents'].fillna(365, inplace=True)
    
    return frequency_metrics

def calculate_abuser_risk_metrics(df):
    """Calculate additional risk metrics for each abuser."""
    risk_metrics = df.groupby('Abuser_ID').agg({
        'Severity_of_Case': ['mean', 'max'],
        'Victim_Outcome': lambda x: (x == 'Hospitalized').mean() * 100,
        'Abuser_Outcome': lambda x: (x == 'Arrested').mean() * 100
    }).reset_index()
    
    risk_metrics.columns = [
        'Abuser_ID', 'avg_severity', 'max_severity',
        'hospitalization_rate', 'arrest_rate'
    ]
    
    return risk_metrics

def main():
    # Load and process data
    df = load_and_prepare_data('incident_data.csv')
    
    # Calculate frequency metrics
    frequency_metrics = calculate_incident_frequency(df)
    
    # Calculate risk metrics
    risk_metrics = calculate_abuser_risk_metrics(df)
    
    # Combine all metrics
    final_metrics = pd.merge(frequency_metrics, risk_metrics, on='Abuser_ID', how='left')
    
    # Convert to integer for readability
    final_metrics['avg_days_between_incidents'] = final_metrics['avg_days_between_incidents'].round().astype(int)
    final_metrics['avg_severity'] = final_metrics['avg_severity'].round().astype(int)
    final_metrics['hospitalization_rate'] = final_metrics['hospitalization_rate'].round().astype(int)
    final_metrics['arrest_rate'] = final_metrics['arrest_rate'].round().astype(int)
    
    # Save results
    final_metrics.to_csv('frequency_analysis.csv', index=False)
    
    return final_metrics

if __name__ == "__main__":
    frequency_analysis = main()
