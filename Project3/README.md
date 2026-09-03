# Customer Segmentation using Unsupervised Learning

## DecodeLab Data Science Internship – Project 3

---

## 📌 Project Overview

This project focuses on **Customer Segmentation using Unsupervised Machine Learning techniques**.

The objective is to discover **hidden customer groups** from retail customer data without using predefined labels.

The project uses:

- **PCA (Principal Component Analysis)**
- **K-Means Clustering**
- **Elbow Method**
- **Silhouette Score**

These techniques are used to identify and analyze different customer segments.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze **customer behavior and purchasing characteristics**
- Clean and prepare the customer dataset
- Standardize numerical features
- Apply **PCA** for dimensionality reduction
- Determine the **optimal number of clusters**
- Apply **K-Means Clustering**
- Visualize customer clusters
- Create meaningful **customer personas**
- Generate actionable **business insights**

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Programming Language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Scikit-learn** | Machine Learning |

---

## 🤖 Machine Learning Techniques

### 1. Data Preprocessing

The customer dataset is cleaned before applying machine learning techniques.

- Missing numerical values are treated using **median imputation**
- Duplicate records are removed
- Relevant numerical features are selected for analysis

---

### 2. Feature Standardization

**StandardScaler** is used to standardize the numerical features.

Standardization ensures that features with different scales contribute appropriately to the clustering process.

---

### 3. Principal Component Analysis (PCA)

**PCA (Principal Component Analysis)** is used for dimensionality reduction.

The high-dimensional customer dataset is reduced to **two principal components**, which makes it easier to visualize customer groups.

---

### 4. Elbow Method

The **Elbow Method** is used to evaluate different values of **K** for K-Means clustering.

It analyzes the **inertia** for different numbers of clusters and helps determine a suitable number of clusters.

---

### 5. Silhouette Score

The **Silhouette Score** is used to evaluate the quality of the clusters.

A higher Silhouette Score indicates better-defined and more separated customer groups.

---

### 6. K-Means Clustering

**K-Means Clustering** is applied using the selected optimal number of clusters.

Customers are grouped based on their similarities in purchasing and behavioral characteristics.

---

## 👥 Customer Personas

The resulting clusters are translated into meaningful business-oriented **customer personas**.

The identified personas include:

- 💎 **Premium High-Value Customers**
- 💰 **Affluent Low-Spending Customers**
- 🛍️ **Budget Enthusiastic Customers**
- 📉 **Low-Engagement Customers**

These personas help businesses understand different types of customers and develop targeted strategies.

---

## 📂 Project Structure

```text
Project3/
│
├── customer_segmentation.py
├── dataset.csv
│
├── output/
│   ├── clustered_customer_data.csv
│   ├── cluster_summary.csv
│   ├── customer_personas.csv
│   └── final_project_summary.txt
│
├── visualizations/
│   ├── 01_pca_visualization.png
│   ├── 02_elbow_method.png
│   ├── 03_silhouette_score.png
│   ├── 04_customer_clusters.png
│   └── 05_customer_personas.png
│
└── README.md
