# DecodeLabs Data Science Internship – Project 1

## Advanced EDA & Feature Engineering

### 📌 Project Overview

This project is completed as part of the **DecodeLabs Data Science Internship – Project 1**.

The main objective of this project is to transform a raw retail orders dataset into a clean and analysis-ready dataset using **Exploratory Data Analysis (EDA), statistical data cleaning, outlier treatment, and feature engineering**.

The project focuses on preparing high-quality data that can be used for further Data Science and Machine Learning tasks.

---

## 🎯 Project Objectives

* Analyze the structure and characteristics of the dataset.
* Identify and handle missing values using statistical imputation.
* Detect and treat outliers using the **Interquartile Range (IQR)** method.
* Create new meaningful features from existing columns.
* Perform exploratory data analysis and visualization.
* Generate a final cleaned dataset ready for further analysis.

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 📊 Dataset

The project uses a **Retail Orders Dataset** containing information such as:

* Order ID
* Order Date
* Category
* Segment
* Region
* Quantity
* Sales
* Discount
* Profit

---

## 🔍 Project Workflow

### 1. Load Dataset

The retail orders CSV file is loaded using Pandas.

### 2. Exploratory Data Analysis

Basic information about the dataset is analyzed using:

* Dataset shape
* `head()`
* `info()`
* `describe()`

### 3. Missing Value Treatment

Missing values in numerical columns are handled using **Median Imputation**.

### 4. Outlier Detection

Outliers are identified using the **IQR (Interquartile Range)** method.

### 5. Outlier Treatment

Detected outliers are treated using **IQR Capping**.

### 6. Feature Engineering

Three new features are created:

* **Profit Margin** – measures profit relative to sales.
* **Discount Amount** – calculates the discount amount based on sales.
* **Sales per Quantity** – calculates sales generated per quantity sold.

### 7. Data Visualization

Different visualizations are generated to understand:

* Sales distribution
* Sales and profit by category
* Sales and profit by region
* Monthly sales and profit trends
* Profit vs. sales relationship
* Feature correlations

### 8. Dashboard

A **Retail Analytics Dashboard** is generated to provide a visual summary of important business insights.

### 9. Save Cleaned Dataset

The final processed dataset is saved as:

`retail_orders_cleaned.csv`

---

## 📈 Output Files

The project generates the following visualization files:

* `fig1_sales_distribution.png`
* `fig2_category_region_analysis.png`
* `fig3_monthly_sales_trend.png`
* `fig4_profit_vs_sales.png`
* `fig5_correlation_heatmap.png`
* `retail_analytics_dashboard.png`

---

## 📁 Project Structure

```text
DecodeLabs_DataScience_Project1/
│
├── project1.py
├── retail-orders-raw.csv
├── retail_orders_cleaned.csv
│
├── fig1_sales_distribution.png
├── fig2_category_region_analysis.png
├── fig3_monthly_sales_trend.png
├── fig4_profit_vs_sales.png
├── fig5_correlation_heatmap.png
│
└── retail_analytics_dashboard.png
```

---

## 🚀 How to Run the Project

### Step 1: Install required libraries

```bash
pip install pandas numpy matplotlib seaborn
```

### Step 2: Keep the dataset in the project folder

Make sure `retail-orders-raw.csv` is available in the same folder as `project1.py`.

### Step 3: Run the Python script

```bash
python project1.py
```

After execution, the cleaned dataset and visualization files will be generated.

---

## ✅ Key Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Statistical Imputation
* Missing Value Handling
* IQR Outlier Detection
* Outlier Treatment
* Feature Engineering
* Data Visualization
* Business-oriented Data Analysis
* Python Programming

---

## 📝 Conclusion

This project provided practical experience in **data wrangling, statistical analysis, EDA, outlier treatment, and feature engineering**.

The raw retail dataset was cleaned and transformed into a structured dataset by handling missing values, treating outliers, and creating meaningful new features. The generated visualizations and dashboard help in understanding sales, profit, category, region, and customer-segment performance.

The final cleaned dataset is now suitable for further **Data Science and Machine Learning analysis**.

---
## 👩‍💻 Internship

**DecodeLabs – Data Science Internship**

**Project:** Project 1 – Advanced EDA & Feature Engineering
