# Customer Churn and Salary Prediction

This project predicts customer churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras. It includes data preprocessing, model training, and a Streamlit web application for real-time predictions. Additionally, it predicts estimated salary using a regression ANN model, with another Streamlit app for salary predictions.

## Dataset

The dataset used is `Churn_Modelling.csv`, which contains customer information such as credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, and estimated salary. For churn prediction, the target variable is `Exited`, indicating whether the customer churned (1) or not (0). For salary prediction, the target variable is `EstimatedSalary`, predicting the customer's estimated salary based on the other features.

## Features

- **Data Preprocessing**: Handles categorical encoding (Label Encoding for Gender, One-Hot Encoding for Geography) and feature scaling.
- **Model Training**: ANN with input layer, two hidden layers (64 and 32 neurons), and sigmoid output for binary classification.
- **Web Application**: Streamlit app for user input and churn prediction.
- **Prediction Notebook**: Jupyter notebook for testing predictions on sample data.
- **Regression Model Training**: ANN with input layer, two hidden layers (64 and 32 neurons), and linear output for salary regression.
- **Salary Prediction Web Application**: Streamlit app for user input and estimated salary prediction.
- **Regression Training Notebook**: Jupyter notebook for salary prediction model training.

## Technologies Used

- Python
- TensorFlow/Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib (for potential visualizations)
- TensorBoard (for model monitoring)
- Keras
- scikeras

## Installation

1. Clone the repository or ensure you have the project files.
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Deployment

The applications are deployed on Streamlit Cloud:

- **Customer Churn Prediction**: [https://ann-classification-bank-churn-abbzx5b89drcqknuarhzdj.streamlit.app/](https://ann-classification-bank-churn-abbzx5b89drcqknuarhzdj.streamlit.app/)
- **Salary Prediction**: [https://ann-bank-balance-prediction-nbwc7gryjd3diooddbnqm8.streamlit.app/](https://ann-bank-balance-prediction-nbwc7gryjd3diooddbnqm8.streamlit.app/)

## Usage

### Running the Web App

To run the Streamlit application for churn prediction:

```
streamlit run app.py
```

This will launch a web interface where you can input customer details and get churn probability.

### Running the Salary Prediction Web App

To run the Streamlit application for salary prediction:

```
streamlit run streamlit_regression.py
```

This will launch a web interface where you can input customer details and get estimated salary.

### Training the Model

Open `experiments.ipynb` in Jupyter Notebook to see the model training process, including data preprocessing, ANN building, and training with early stopping and TensorBoard logging.

### Training the Regression Model

Open `salaryregression.ipynb` in Jupyter Notebook to see the salary prediction model training process, including data preprocessing, ANN building, and training with early stopping and TensorBoard logging.

### Making Predictions

Use `prediction.ipynb` to load the trained model and make predictions on sample data.

## Files Description

- `app.py`: Streamlit web application for churn prediction.
- `experiments.ipynb`: Jupyter notebook for data preprocessing and model training.
- `prediction.ipynb`: Jupyter notebook for making predictions on example data.
- `Churn_Modelling.csv`: Dataset file.
- `model.keras`: Trained ANN model.
- `scaler.pkl`: StandardScaler for feature scaling.
- `label_encoder_gender.pkl`: LabelEncoder for Gender.
- `onehot_encoder_geo.pkl`: OneHotEncoder for Geography.
- `requirements.txt`: List of Python dependencies.
- `logs/`: Directory for TensorBoard logs.
- `salaryregression.ipynb`: Jupyter notebook for salary prediction model training.
- `streamlit_regression.py`: Streamlit web application for salary prediction.
- `regression_model.keras`: Trained ANN regression model.
- `scaler1.pkl`: StandardScaler for feature scaling (regression).
- `label_encoder_gender1.pkl`: LabelEncoder for Gender (regression).
- `onehot_encoder_geo1.pkl`: OneHotEncoder for Geography (regression).
- `regressionlogs/`: Directory for TensorBoard logs (regression).

## Model Performance

The churn prediction model is trained with Adam optimizer, binary cross-entropy loss, and accuracy metric. Early stopping is implemented to prevent overfitting. The salary prediction model is trained with Adam optimizer, mean absolute error (MAE) loss, and MAE metric. Early stopping is also implemented for the regression model.

## License

This project is open-source. Please check for any specific licensing requirements.
