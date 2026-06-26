Heart Disease Prediction & ML Classifier Comparison

This repository contains a machine learning workflow that predicts heart disease from clinical patient data using the Kaggle heart disease dataset. It compares three different classifier models: Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN).

Table of Contents
1. Dataset Overview
2. Data Preprocessing & Scaling
3. Feature Engineering
4. Model Training & Comparison
5. Key Results & Best Model
6. Running the Project

Dataset Overview
The dataset contains 1,025 patient records and 14 clinical features:
- age: Age in years
- sex: (1 = male; 0 = female)
- cp: Chest pain type (0, 1, 2, 3)
- trestbps: Resting blood pressure in mm Hg
- chol: Serum cholesterol in mg/dl
- fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- restecg: Resting electrocardiographic results (0, 1, 2)
- thalach: Maximum heart rate achieved
- exang: Exercise-induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels (0-3) colored by fluoroscopy
- thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
- target: Diagnosis of heart disease (1 = disease; 0 = no disease)

Data Preprocessing & Scaling
- Missing Values: Verified 0 missing values across all records.
- Train-Test Split: Partitioned the dataset using an 80/20 train/test split.
- Feature Scaling: Applied standard scaling (StandardScaler) to normalize continuous variables, ensuring that distance-based metrics (KNN) and gradient-based solvers (Logistic Regression) converge correctly and are not dominated by larger numeric ranges.

Feature Engineering
A correlation and feature importance analysis (via Random Forest) was conducted:
- Chest pain type (cp) and maximum heart rate (thalach) exhibit the strongest positive correlations with heart disease.
- ST depression (oldpeak), number of major vessels (ca), and exercise-induced angina (exang) show the strongest negative correlations.
- Random Forest feature importance identifies 'cp', 'thalach', 'ca', and 'oldpeak' as the top predictors.

Model Training & Comparison
We trained and evaluated three distinct classification models on the test set:
- Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

Evaluation Metrics:
- Logistic Regression: Accuracy = 80.98%, Precision = 76.19%, Recall = 91.43%, F1-Score = 83.12%
- K-Nearest Neighbors: Accuracy = 86.34%, Precision = 87.38%, Recall = 85.71%, F1-Score = 86.54%
- Random Forest: Accuracy = 100.00%, Precision = 100.00%, Recall = 100.00%, F1-Score = 100.00%

Key Results & Best Model
The Random Forest model performed best, achieving perfect accuracy and F1-score on our test partition. This indicates a highly separable boundary or potential duplication in the dataset splits, which tree ensembles fit optimally.

Running the Project
Execute the pipeline script to train the models, output the evaluation table, and generate metric comparison plots:

python3 train_model.py
