import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

def prepare_features(df, frequency_metrics):
    """Prepare features for the prediction model."""
    # Merge with frequency metrics
    df = pd.merge(df, frequency_metrics, on='Abuser_ID', how='left')
    
    # Calculate abuser age at incident
    df['abuser_age'] = (df['Incident_Date'] - df['DOB_of_Abuser']).dt.days / 365
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Type_of_Incident_encoded'] = le.fit_transform(df['Type_of_Incident'])
    df['Relationship_encoded'] = le.fit_transform(df['Relationship_to_Victim'])
    df['Oblast_encoded'] = le.fit_transform(df['Oblast'])
    
    # Create feature matrix
    features = [
        'incident_count', 'avg_days_between_incidents', 'avg_severity',
        'max_severity', 'hospitalization_rate', 'arrest_rate',
        'abuser_age', 'Type_of_Incident_encoded', 'Relationship_encoded',
        'Oblast_encoded'
    ]
    
    return df[features]

def train_prediction_model(df, frequency_metrics):
    """Train a model to predict days until next incident."""
    # Prepare features
    X = prepare_features(df, frequency_metrics)
    
    # Target variable: ensure alignment by merging `avg_days_between_incidents` with `df`
    df_merged = pd.merge(df[['Abuser_ID']], frequency_metrics[['Abuser_ID', 'avg_days_between_incidents']], on='Abuser_ID', how='left')
    y = df_merged['avg_days_between_incidents']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


def predict_next_incident(model, df, frequency_metrics):
    """Predict the next incident date for each abuser."""
    # Prepare features for prediction, keeping only one row per Abuser_ID
    df_unique = df.drop_duplicates(subset='Abuser_ID')
    X_pred = prepare_features(df_unique, frequency_metrics)
    
    # Make predictions
    days_until_next = model.predict(X_pred)
    
    # Ensure correct number of predictions
    unique_abusers = df_unique['Abuser_ID'].values
    if len(unique_abusers) != len(days_until_next):
        raise ValueError("Mismatch between number of unique Abuser_IDs and prediction length.")
    
    # Get the latest incident date for each abuser
    latest_incidents = df.groupby('Abuser_ID')['Incident_Date'].max().reset_index()
    
    # Calculate next incident dates
    predictions = pd.DataFrame({
        'Abuser_ID': unique_abusers,
        'days_until_next': days_until_next
    })
    
    predictions = pd.merge(predictions, latest_incidents, on='Abuser_ID', how='left')
    
    # Convert 'days_until_next' to timedelta and add to the latest incident date
    predictions['next_incident_date'] = pd.to_datetime(predictions['Incident_Date']) + \
        pd.to_timedelta(predictions['days_until_next'], unit='D')
    
    # Explicitly convert 'next_incident_date' to datetime to ensure the correct format
    predictions['next_incident_date'] = pd.to_datetime(predictions['next_incident_date'])
    
    # Ensure 'days_until_next' is rounded and converted to an integer
    predictions['days_until_next'] = predictions['days_until_next'].round().astype(int)
    
    # Check if predictions is None
    if predictions is None:
        print("Predictions dataframe is None.")
    else:
        print(f"Predictions dataframe has {len(predictions)} rows.")
    
    # Check the data types
    print(predictions.dtypes)  # This will help check the types of the columns
    
    return predictions[['Abuser_ID', 'next_incident_date', 'days_until_next']]




def main():
    # Load data
    df = pd.read_csv('incident_data.csv')
    df['Incident_Date'] = pd.to_datetime(df['Incident_Date'])
    df['DOB_of_Abuser'] = pd.to_datetime(df['DOB_of_Abuser'])
    
    # Load frequency metrics
    frequency_metrics = pd.read_csv('frequency_analysis.csv')
    
    # Train model
    model, X_test, y_test = train_prediction_model(df, frequency_metrics)
    
    # Make predictions
    predictions = predict_next_incident(model, df, frequency_metrics)
    
    # Check if predictions are valid
    if predictions is None:
        print("Predictions were not generated.")
    else:
        print(f"Predictions generated successfully with {len(predictions)} rows.")
    
    # Save predictions
    if predictions is not None:
        predictions.to_csv('incident_predictions.csv', index=False, date_format='%Y-%m-%d')
    else:
        print("Skipping saving predictions because they are None.")
    
    return predictions, model

if __name__ == "__main__":
    predictions, model = main()

