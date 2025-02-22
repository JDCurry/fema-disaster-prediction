# src/main.py

from src.predictor import DisasterPredictor  # Update import path

def main():
    # Initialize predictor
    predictor = DisasterPredictor('data/fema_disasters_complete.csv')
    
    # Train models
    predictor.train_type_predictor()
    predictor.train_duration_predictor()
    
    # Save models
    predictor.save_models()
    
    # Make predictions
    predictions = predictor.predict_future_events(future_days=90)
    
    # Plot results
    predictor.plot_predictions(predictions)
    
    # Save high-risk predictions
    high_risk_predictions = predictions[predictions['probability'] > 0.4]
    high_risk_predictions.to_csv('data/disaster_predictions.csv', index=False)
    print("\nHigh-risk predictions saved to 'data/disaster_predictions.csv'")
    
    # Display summary
    print("\nHighest Risk Predictions:")
    high_risk_summary = high_risk_predictions.groupby('disaster_type').agg({
        'probability': ['mean', 'max'],
        'predicted_duration': 'mean'
    }).round(2)
    print(high_risk_summary)

if __name__ == "__main__":
    main()