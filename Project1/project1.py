import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("retail-orders-raw.csv")

print("\n===== DATASET LOADED =====")
print(df.head())


# ==========================================
# STEP 2: BASIC INFORMATION
# ==========================================

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATASET INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())


# ==========================================
# STEP 3: CHECK MISSING VALUES
# ==========================================

print("\n===== MISSING VALUES BEFORE TREATMENT =====")
print(df.isnull().sum())


# ==========================================
# STEP 4: HANDLE MISSING VALUES
# Using Median Imputation
# ==========================================

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\n===== MISSING VALUES AFTER TREATMENT =====")
print(df.isnull().sum())


# ==========================================
# STEP 5: OUTLIER DETECTION USING IQR
# ==========================================

print("\n===== OUTLIER DETECTION =====")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"{column}: {len(outliers)} outliers")


# ==========================================
# STEP 6: OUTLIER TREATMENT
# Using IQR CAPPING
# ==========================================

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower_limit,
        upper_limit
    )

print("\n===== OUTLIERS TREATED =====")


# ==========================================
# STEP 7: FEATURE ENGINEERING
# Create 3 New Features
# ==========================================

# Feature 1: Profit Margin
df["Profit_Margin"] = (
    df["Profit"] / df["Sales"]
)

# Feature 2: Discount Amount
df["Discount_Amount"] = (
    df["Sales"] * df["Discount"]
)

# Feature 3: Sales per Quantity
df["Sales_per_Quantity"] = (
    df["Sales"] / df["Quantity"]
)

print("\n===== NEW FEATURES CREATED =====")
print(
    df[
        [
            "Profit_Margin",
            "Discount_Amount",
            "Sales_per_Quantity"
        ]
    ].head()
)


# ==========================================
# STEP 8: CHECK FINAL DATASET
# ==========================================

print("\n===== FINAL DATASET =====")
print(df.head())

print("\n===== FINAL DATASET INFORMATION =====")
df.info()

print("\n===== FINAL MISSING VALUES =====")
print(df.isnull().sum())


# ==========================================
# STEP 9: VISUALIZATION & ANALYTICAL DASHBOARD
# ==========================================

# Set styling theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# Ensure Order_Date is datetime
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\n===== GENERATING VISUALIZATIONS =====")

# 1. Sales Distribution & Boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Sales"], kde=True, ax=axes[0], color="#2b5c8f")
axes[0].set_title("Sales Distribution (KDE)", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Sales ($)")
axes[0].set_ylabel("Frequency")

sns.boxplot(x=df["Sales"], ax=axes[1], color="#e07a5f")
axes[1].set_title("Sales Outlier Boxplot", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Sales ($)")
plt.savefig("fig1_sales_distribution.png", dpi=300)
plt.close()
print(" -> Saved: fig1_sales_distribution.png")

# 2. Sales & Profit by Category and Region
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cat_perf = df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
cat_perf_melted = cat_perf.melt(id_vars="Category", value_vars=["Sales", "Profit"], var_name="Metric", value_name="Amount")
sns.barplot(data=cat_perf_melted, x="Category", y="Amount", hue="Metric", ax=axes[0], palette="Blues_d")
axes[0].set_title("Total Sales & Profit by Category", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Amount ($)")

region_perf = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index()
region_perf_melted = region_perf.melt(id_vars="Region", value_vars=["Sales", "Profit"], var_name="Metric", value_name="Amount")
sns.barplot(data=region_perf_melted, x="Region", y="Amount", hue="Metric", ax=axes[1], palette="Set2")
axes[1].set_title("Total Sales & Profit by Region", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Amount ($)")
plt.savefig("fig2_category_region_analysis.png", dpi=300)
plt.close()
print(" -> Saved: fig2_category_region_analysis.png")

# 3. Monthly Sales & Profit Trend Over Time
if "Order_Date" in df.columns:
    df_monthly = df.set_index("Order_Date").resample("ME")[["Sales", "Profit"]].sum().reset_index()
    plt.figure(figsize=(12, 5))
    plt.plot(df_monthly["Order_Date"], df_monthly["Sales"], marker='o', linewidth=2.5, label="Sales", color="#1f77b4")
    plt.plot(df_monthly["Order_Date"], df_monthly["Profit"], marker='s', linewidth=2.5, label="Profit", color="#2ca02c")
    plt.title("Monthly Sales & Profit Trend", fontsize=14, fontweight='bold')
    plt.xlabel("Order Month")
    plt.ylabel("Amount ($)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("fig3_monthly_sales_trend.png", dpi=300)
    plt.close()
    print(" -> Saved: fig3_monthly_sales_trend.png")

# 4. Profit vs. Sales Scatter Plot
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Category", style="Segment", size="Quantity", sizes=(30, 200), alpha=0.8)
plt.title("Profit vs. Sales by Category & Segment", fontsize=14, fontweight='bold')
plt.xlabel("Sales ($)")
plt.ylabel("Profit ($)")
plt.savefig("fig4_profit_vs_sales.png", dpi=300)
plt.close()
print(" -> Saved: fig4_profit_vs_sales.png")

# 5. Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.savefig("fig5_correlation_heatmap.png", dpi=300)
plt.close()
print(" -> Saved: fig5_correlation_heatmap.png")

# 6. Comprehensive Retail Analytics Dashboard Grid
fig = plt.figure(figsize=(16, 12))
fig.suptitle("RETAIL ORDERS DATA SCIENCE DASHBOARD", fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Category Sales
ax1 = fig.add_subplot(2, 3, 1)
cat_sales = df.groupby("Category")["Sales"].sum()
ax1.pie(cat_sales, labels=cat_sales.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
ax1.set_title("Sales Share by Category", fontweight='bold')

# Subplot 2: Segment Sales
ax2 = fig.add_subplot(2, 3, 2)
seg_sales = df.groupby("Segment")["Sales"].sum().reset_index()
sns.barplot(data=seg_sales, x="Segment", y="Sales", ax=ax2, palette="crest")
ax2.set_title("Sales by Customer Segment", fontweight='bold')

# Subplot 3: Sales Distribution
ax3 = fig.add_subplot(2, 3, 3)
sns.histplot(df["Sales"], kde=True, ax=ax3, color="#4c72b0")
ax3.set_title("Sales Distribution", fontweight='bold')

# Subplot 4: Profit Margin Distribution
ax4 = fig.add_subplot(2, 3, 4)
sns.boxplot(data=df, x="Category", y="Profit_Margin", ax=ax4, palette="Set3")
ax4.set_title("Profit Margin by Category", fontweight='bold')

# Subplot 5: Region Performance
ax5 = fig.add_subplot(2, 3, 5)
sns.barplot(data=region_perf_melted, x="Region", y="Amount", hue="Metric", ax=ax5, palette="viridis")
ax5.set_title("Region Performance", fontweight='bold')

# Subplot 6: Heatmap
ax6 = fig.add_subplot(2, 3, 6)
sns.heatmap(correlation, annot=True, fmt=".1f", cmap="mako", ax=ax6, cbar=False)
ax6.set_title("Correlation Matrix", fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("retail_analytics_dashboard.png", dpi=300)
plt.close()
print(" -> Saved: retail_analytics_dashboard.png")


# ==========================================
# STEP 10: SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    "retail_orders_cleaned.csv",
    index=False
)

print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("Cleaned dataset saved as:")
print("retail_orders_cleaned.csv")
print("================================")