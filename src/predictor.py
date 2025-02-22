# src/predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from .utils import (
    load_data,
    extract_temporal_features,
    calculate_seasonal_risk,
    generate_future_dates
)

class DisasterPredictor:
    def __init__(self, csv_path):
        # Load and prepare data
        self.data = load_data(csv_path)
        self.data = extract_temporal_features(self.data)
        self.data['seasonal_risk'] = calculate_seasonal_risk(self.data)
        self.label_encoders = {}
        
    def train_type_predictor(self):
        """Train a model to predict disaster type"""
        features = ['month', 'day_of_year', 'region', 'seasonal_risk', 'fipsStateCode']
        X = self.data[features].copy()
        y = self.data['incidentType']
        
        # Handle categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.type_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.type_model.fit(X_train, y_train)
        
        y_pred = self.type_model.predict(X_test)
        print("\nDisaster Type Prediction Performance:")
        print(classification_report(y_test, y_pred))
        
        return self.type_model
        
    def train_duration_predictor(self):
        """Train a model to predict disaster duration"""
        features = ['month', 'day_of_year', 'region', 'seasonal_risk', 'fipsStateCode']
        X = self.data[features].copy()
        y = self.data['incident_duration']
        
        # Remove any NaN values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.duration_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.duration_model.fit(X_train, y_train)
        
        y_pred = self.duration_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"\nDuration Prediction MSE: {mse:.2f} days")
        
        return self.duration_model
        
    def predict_future_events(self, future_days=90):
        """Predict likely disasters for the next X days"""
        future_df = generate_future_dates(future_days)
        regions = self.data['region'].unique()
        predictions = []
        
        for region in regions:
            region_data = future_df.copy()
            region_data['region'] = region
            region_data['fipsStateCode'] = self.data[self.data['region'] == region]['fipsStateCode'].mode()[0]
            region_data['seasonal_risk'] = np.mean(
                self.data[self.data['region'] == region]['seasonal_risk']
            )
            
            X_pred = region_data[['month', 'day_of_year', 'region', 'seasonal_risk', 'fipsStateCode']]
            
            type_probs = self.type_model.predict_proba(X_pred)
            predicted_types = self.type_model.classes_
            durations = self.duration_model.predict(X_pred)
            
            for i, date in enumerate(future_df['date']):
                for j, disaster_type in enumerate(predicted_types):
                    if type_probs[i][j] > 0.2:
                        predictions.append({
                            'date': date,
                            'region': region,
                            'disaster_type': disaster_type,
                            'probability': type_probs[i][j],
                            'predicted_duration': durations[i]
                        })
        
        return pd.DataFrame(predictions)
        
    def plot_predictions(self, predictions_df):
        """Visualize predictions"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Probability heatmap
        plt.subplot(2, 1, 1)
        pivot_data = predictions_df.pivot_table(
            values='probability',
            index='region',
            columns='disaster_type',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Predicted Disaster Probabilities by Region')
        
        # Plot 2: Timeline
        plt.subplot(2, 1, 2)
        for disaster_type in predictions_df['disaster_type'].unique():
            type_data = predictions_df[predictions_df['disaster_type'] == disaster_type]
            plt.scatter(type_data['date'], type_data['probability'],
                       label=disaster_type, alpha=0.6)
        
        plt.xlabel('Date')
        plt.ylabel('Probability')
        plt.title('Predicted Disasters Timeline')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def save_models(self, path='models'):
        """Save trained models"""
        joblib.dump(self.type_model, f'{path}/type_model.joblib')
        joblib.dump(self.duration_model, f'{path}/duration_model.joblib')
        joblib.dump(self.label_encoders, f'{path}/label_encoders.joblib')
        
    def load_models(self, path='models'):
        """Load trained models"""
        self.type_model = joblib.load(f'{path}/type_model.joblib')
        self.duration_model = joblib.load(f'{path}/duration_model.joblib')
        self.label_encoders = joblib.load(f'{path}/label_encoders.joblib')