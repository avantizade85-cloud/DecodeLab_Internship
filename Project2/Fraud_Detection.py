import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE



# FOLDERS & OUTPUT FILE

os.makedirs("output", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


output_file = open("output/project_output.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, output_file)


print("\n========================================")
print("FRAUD DETECTION PROJECT 2")
print("========================================")



# LOAD & EXPLORE DATA


df = pd.read_csv("data/raw/fraud_dataset.csv")

print("\nDataset loaded successfully!")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Duplicate Rows:", df.duplicated().sum())



# FRAUD DISTRIBUTION


print("\n========== FRAUD DISTRIBUTION ==========")
print(df["Fraud"].value_counts())

print("\nFraud Percentage:")
print(df["Fraud"].value_counts(normalize=True) * 100)


plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Fraud")
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Fraud (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("visualizations/fraud_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# FEATURES & ENCODING


X = df.drop(columns=["Transaction_ID", "Fraud"])
y = df["Fraud"]

print("\n========== FEATURES ==========")
print(X.head())

print("\n========== TARGET ==========")
print(y.head())

X = pd.get_dummies(
    X,
    columns=["Location", "Transaction_Type"],
    drop_first=True
)

print("\n========== ENCODED FEATURES ==========")
print(X.head())

print("\nFeature columns:")
print(X.columns.tolist())



# TRAIN TEST SPLIT & SCALING


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")



# SMOTE

print("\n========== BEFORE SMOTE ==========")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled, y_train
)

print("\n========== AFTER SMOTE ==========")
print(pd.Series(y_train_smote).value_counts())


plt.figure(figsize=(7, 5))
sns.countplot(x=y_train_smote)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Fraud Class")
plt.ylabel("Number of Samples")
plt.savefig("visualizations/smote_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# LOGISTIC REGRESSION


print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_smote, y_train_smote)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_roc_auc = roc_auc_score(y_test, lr_prob)

print("\nPrecision:")
print(lr_precision)

print("\nRecall:")
print(lr_recall)

print("\nROC-AUC:")
print(lr_roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, lr_pred, zero_division=0))


# LOGISTIC REGRESSION CONFUSION MATRIX


cm_lr = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt="d")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("visualizations/confusion_matrix_lr.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()



# RANDOM FOREST

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_smote, y_train_smote)

rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\nPrecision:")
print(rf_precision)

print("\nRecall:")
print(rf_recall)

print("\nROC-AUC:")
print(rf_roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred, zero_division=0))



# RANDOM FOREST CONFUSION MATRIX

cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("visualizations/confusion_matrix_rf.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()



# ROC CURVE


lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr, lr_tpr,
    label=f"Logistic Regression (AUC = {lr_roc_auc:.3f})"
)

plt.plot(
    rf_fpr, rf_tpr,
    label=f"Random Forest (AUC = {rf_roc_auc:.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("visualizations/roc_curve.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# MODEL COMPARISON


results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Precision": [lr_precision, rf_precision],
    "Recall": [lr_recall, rf_recall],
    "ROC-AUC": [lr_roc_auc, rf_roc_auc]
})

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results)

results.to_csv("output/model_results.csv", index=False)

print("\nModel results saved successfully!")


# MODEL COMPARISON VISUALIZATION


results.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()

plt.savefig("visualizations/model_comparison.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# FEATURE IMPORTANCE


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("TOP 10 IMPORTANT FEATURES")
print("========================================")

print(feature_importance.head(10))


plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importance - Random Forest")

plt.savefig("visualizations/feature_importance.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

feature_importance.to_csv(
    "output/feature_importance.csv",
    index=False
)


# SAVE DATA & PREDICTIONS


df.to_csv(
    "output/fraud_dataset_cleaned.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")


predictions = pd.DataFrame({
    "Actual_Fraud": y_test.values,
    "Logistic_Regression_Prediction": lr_pred,
    "Random_Forest_Prediction": rf_pred
})

predictions.to_csv(
    "output/model_predictions.csv",
    index=False
)

print("Model predictions saved successfully!")


# FINAL OUTPUT


best_model = results.loc[
    results["ROC-AUC"].idxmax(),
    "Model"
]

print("\n========================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nFinal Model Results:")
print(results)

print("\nBest ROC-AUC Model:")
print(best_model)

print("\nTerminal output saved in:")
print("output/project_output.txt")

print("\nProject output files saved in:")
print("output/")

print("\nVisualizations saved in:")
print("visualizations/")


output_file.close()