# 🏠 House Price Prediction
A machine learning project to predict house prices using various regression models (Linear, Ridge, Lasso). The project includes full data preprocessing, feature engineering, model selection, cross-validation, and model evaluation.

 link="https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
![House Prices Banner](https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png)


---

## 📌 Project Overview

This project builds a complete ML pipeline for predicting house prices using the Ames Housing dataset.  
The workflow includes:

- Data loading and cleaning
- Handling missing values and outliers
- Feature encoding and scaling
- Log transformation for skewed features
- Model training and selection (Ridge, Lasso, Linear Regression)
- Model evaluation using R²
- Saving the final model 

---

## Model Comparison

| Model           | R² Score |
|----------------|----------|
| Linear         | 0.8663    | 
| Ridge          | 0.868     | 
| Lasso          | 0.866     | 

---

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/moalli99/house_price_prediction.git
cd house_price_prediction

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py
