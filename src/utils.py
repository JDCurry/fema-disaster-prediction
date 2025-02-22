# src/utils.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(csv_path):
    """Load and perform initial data cleaning"""
    df = pd.read_csv(csv_path)
    
    # Convert dates
    date_columns = ['declarationDate', 'incidentBeginDate', 'incidentEndDate']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
        
    return df

def extract_temporal_features(df):
    """Extract temporal features from the dataset"""
    df['month'] = df['incidentBeginDate'].dt.month
    df['year'] = df['incidentBeginDate'].dt.year
    df['day_of_year'] = df['incidentBeginDate'].dt.dayofyear
    
    # Calculate incident duration if not present
    if 'incident_duration' not in df.columns:
        df['incident_duration'] = (
            df['incidentEndDate'] - df['incidentBeginDate']
        ).dt.total_seconds() / (24 * 3600)  # Convert to days
        
    return df

def calculate_seasonal_risk(df):
    """Calculate seasonal risk scores based on historical patterns"""
    risk_scores = np.zeros(len(df))
    
    for region in df['region'].unique():
        for incident_type in df['incidentType'].unique():
            mask = (df['region'] == region) & (df['incidentType'] == incident_type)
            monthly_counts = df[mask].groupby('month').size()
            
            if not monthly_counts.empty:
                normalized_counts = (monthly_counts - monthly_counts.min()) / \
                                 (monthly_counts.max() - monthly_counts.min() + 1e-6)
                
                for month, score in normalized_counts.items():
                    month_mask = mask & (df['month'] == month)
                    risk_scores[month_mask] = score
                    
    return risk_scores

def generate_future_dates(days=90):
    """Generate future dates for prediction"""
    current_date = datetime.now()
    future_dates = [current_date + timedelta(days=x) for x in range(days)]
    
    return pd.DataFrame({
        'date': future_dates,
        'month': [d.month for d in future_dates],
        'day_of_year': [d.timetuple().tm_yday for d in future_dates],
    })