# Exam Score Prediction using Machine Learning

This project predicts students' math exam scores based on their test preparation course status and scores in reading and writing using machine learning models.

## Description

The dataset contains students' scores and their test preparation course completion status. We use this data to train regression models to predict math scores. The project compares the performance of:

- **Linear Regression**
- **Random Forest Regressor**

Various evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score are used to assess model performance.

## Dataset

The dataset `exams.csv` includes the following columns:

- `test preparation course`: whether the student completed a prep course (`none` or `completed`)
- `math score`
- `reading score`
- `writing score`

## How to Run

1. Ensure you have Python 3 installed.
2. Install required libraries: pandas, numpy, scikit-learn, matplotlib.
3. Run the script to train models and evaluate predictions.

```bash
python model.py
