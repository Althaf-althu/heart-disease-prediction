import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("=== Step 1: Load, Explore & Preprocess ===")
    # Load dataset
    data_path = "heart.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing values check:")
    print(df.isnull().sum())
    
    print("\nData Types:")
    print(df.dtypes)
    
    # We inspect the target distribution
    print("\nTarget Class Distribution:")
    print(df['target'].value_counts(normalize=True))
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split train/test sets (80/20) with a fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Scale features (required for KNN and Logistic Regression to perform optimally)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled features back to DataFrames for easier handling in feature analysis
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("\n=== Step 2: Feature Engineering & Correlation Analysis ===")
    # Correlation analysis
    correlations = df.corr()['target'].sort_values(ascending=False)
    print("Correlation of features with target:")
    print(correlations)
    
    # Train a temporary Random Forest to get feature importances
    rf_temp = RandomForestClassifier(random_state=42)
    rf_temp.fit(X_train, y_train)
    importances = pd.Series(rf_temp.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nRandom Forest Feature Importances:")
    print(importances)
    
    # Let's save a feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'.")
    
    print("\n=== Step 3: Train 3 Different Models ===")
    # 1.Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled_df, y_train)
    
    # 2.Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_scaled_df, y_train)
    
    # 3.K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled_df, y_train)
    
    print("Successfully trained: Logistic Regression, Random Forest, and KNN.")
    
    print("\n=== Step 4: Evaluate & Compare All Models ===")
    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "K-Nearest Neighbors": knn
    }
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled_df)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Save the table to a CSV file for report reference
    results_df.to_csv("model_comparison.csv", index=False)
    
    # Plot metric comparison across models
    df_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Value", hue="Model", palette="Set2")
    plt.title("Model Comparison by Metric")
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png")
    plt.close()
    print("Model comparison plot saved as 'model_comparison_metrics.png'.")
    
    # Determine the best model based on F1-Score (balances precision and recall)
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_model = models[best_model_name]
    print(f"\nBest Performing Model: {best_model_name}")
    
    print("\n=== Step 5: Best Model Analysis & Confusion Matrix ===")
    y_pred_best = best_model.predict(X_test_scaled_df)
    cm = confusion_matrix(y_test, y_pred_best)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease (0)', 'Disease (1)'], 
                yticklabels=['No Disease (0)', 'Disease (1)'])
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")

if __name__ == "__main__":
    main()
