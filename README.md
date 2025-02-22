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
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request