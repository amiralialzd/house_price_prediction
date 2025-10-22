# House Price Prediction

This project is an **end-to-end machine learning pipeline** to predict house prices using the Kaggle "House Prices: Advanced Regression Techniques" dataset.

## 🚀 Project Overview
The goal of this project is to build a model that accurately predicts house sale prices based on key features. The pipeline includes:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering (One-Hot Encoding, handling missing values)  
- Model Training (Linear Regression, Polynomial Regression, Ridge Regression)  
- Hyperparameter Tuning using GridSearchCV  
- Cross-Validation for model evaluation  
- Generating predictions on Kaggle test dataset  

## 📂 Dataset
- **train.csv** – contains features and target variable (`SalePrice`)  
- **test.csv** – contains features only (for generating predictions for Kaggle submission)  

## 🛠 Tools & Libraries
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (LinearRegression, Ridge, GridSearchCV, Pipeline, PolynomialFeatures, StandardScaler)  

## 📊 Key Features Used
- `OverallQual` – Overall quality of the house  
- `GrLivArea` – Above grade (ground) living area square feet  
- `GarageArea` – Size of garage in square feet  
- `Neighborhood` – Location of the house (one-hot encoded)  

## 📈 Modeling Approach
1. Handle missing values (replace with mean/mode or 'unknown')  
2. One-Hot Encode categorical variables  
3. Train Linear Regression and Ridge Regression models  
4. Hyperparameter tuning using GridSearchCV for Ridge  
5. Evaluate models using R² and cross-validation  
6. Make predictions on test set and save `submission.csv`  

## 🏆 Results
- Best model: **Ridge Regression with GridSearchCV**  
- Cross-validated R²: ~0.719  

## 📂 Files
- `house_price_prediction.py` – Main Python script  
- `submission.csv` – Kaggle submission file  
- `train.csv`, `test.csv` – Dataset files  

## 📌 How to Run
1. Clone this repository  
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn


    
