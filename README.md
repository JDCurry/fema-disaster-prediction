# FEMA Disaster Prediction Model

A machine learning model for predicting natural disasters based on historical FEMA data.

## Overview

This project uses historical FEMA disaster declaration data to predict potential future disasters. It includes:
- Disaster type prediction
- Duration prediction
- Seasonal risk analysis
- Regional probability assessment

## Features

- Temporal pattern analysis
- Geographic distribution analysis
- Seasonal risk scoring
- Multi-model prediction system
- Visualization tools

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDCurry/fema-disaster-prediction/blob/main/notebooks/FEMA_Disaster_Analysis.ipynb)


1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fema-disaster-prediction.git
   cd fema-disaster-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your FEMA disaster data in the `data/` directory.

2. Run the prediction model:
   ```bash
   python src/predictor.py
   ```

3. Check the output in the `models/` directory and view the visualizations.

## Data Sources

The model uses FEMA Disaster Declarations Summaries dataset, which can be obtained from:
https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries

## Model Details

The prediction system uses:
- Random Forest Classifier for disaster type prediction
- Random Forest Regressor for duration prediction
- Custom seasonal risk scoring algorithm

## License

MIT License

## Contributing

1. Fork the repository
3. Create a feature branch
4. Commit your changes
5. Push to the branch
6. Create a Pull Request

## Future Updates/Improvements
Current Prediction Models and Data Sources

Disaster Type Prediction:
Uses a Random Forest Classifier to predict the type of disaster.
Implementation: src/predictor.py

Disaster Duration Prediction:
Uses a Random Forest Regressor to predict the duration of a disaster.
Implementation: src/predictor.py

Future Event Prediction:
Predicts likely disasters for the next X days using historical data.
Implementation: src/predictor.py

Data Sources:
The model uses FEMA Disaster Declarations Summaries dataset.
Data sources are listed in the README.md.

Potential Improvements

Advanced Modeling Techniques:
Explore Gradient Boosting Machines (GBM), XGBoost, or LightGBM for better performance.
Use deep learning models like LSTM or GRU for time series prediction.

Feature Engineering:
Incorporate additional features such as weather data, economic factors, or population density.
Use feature selection techniques to identify the most relevant features.

Ensemble Methods:
Combine multiple models to create an ensemble model for better accuracy.
Use stacking, bagging, or boosting techniques.

Hyperparameter Tuning:
Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV to find the best parameters for your models.

Cross-Validation:
Use cross-validation techniques to ensure the robustness of your models.

Additional Data Sources:
Integrate more diverse datasets to improve the prediction capability.
Implement and test some of these improvements to measure the impact on prediction accuracy.
