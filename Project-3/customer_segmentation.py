import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("customer_segmentation_dataset.csv")

print("========================================")
print("   CUSTOMER SEGMENTATION PROJECT")
print("========================================")

print("\nDataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print("Duplicates:", df.duplicated().sum())
# ========================================
# STEP 8 - DATA CLEANING
# ========================================

print("\n===== DATA CLEANING =====")

# Check missing values before treatment
print("\nMissing values before treatment:")
print(df.isnull().sum())

# Select numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

# Fill missing values with median
df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values after treatment
print("\nMissing values after treatment:")
print(df.isnull().sum())

print("\nDuplicate rows after treatment:")
print(df.duplicated().sum())

print("\nData cleaning completed successfully!")
# ========================================
# STEP 9 - FEATURE SELECTION
# ========================================

print("\n===== FEATURE SELECTION =====")

# Remove Customer_ID because it is an identifier
features = df.drop(columns=["Customer_ID"])

# Select numerical features
features = features.select_dtypes(include=np.number)

print("\nFeatures selected for clustering:")
print(features.columns.tolist())

print("\nNumber of features:", features.shape[1])
print("Feature data shape:", features.shape)
# ========================================
# STEP 10 - FEATURE STANDARDIZATION
# ========================================

from sklearn.preprocessing import StandardScaler

print("\n===== FEATURE STANDARDIZATION =====")

# Create StandardScaler
scaler = StandardScaler()

# Standardize all selected features
scaled_data = scaler.fit_transform(features)

print("Standardization completed successfully!")

print("\nOriginal data shape:", features.shape)
print("Scaled data shape:", scaled_data.shape)

print("\nFirst 5 rows of scaled data:")
print(scaled_data[:5])
# ========================================
# STEP 11 - PRINCIPAL COMPONENT ANALYSIS
# ========================================

from sklearn.decomposition import PCA

print("\n===== PRINCIPAL COMPONENT ANALYSIS =====")

# Reduce 23 dimensions to 2 dimensions
pca = PCA(n_components=2)

pca_data = pca.fit_transform(scaled_data)

print("PCA completed successfully!")

print("\nOriginal dimensions:", scaled_data.shape[1])
print("Reduced dimensions:", pca_data.shape[1])

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal explained variance:",
    pca.explained_variance_ratio_.sum()
)

print("\nPCA data shape:", pca_data.shape)
# ========================================
# STEP 12 - PCA VISUALIZATION
# ========================================

import matplotlib.pyplot as plt

print("\n===== PCA VISUALIZATION =====")

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Data After PCA")

plt.tight_layout()

plt.savefig(
    "visualizations/01_pca_visualization.png",
    dpi=300
)

plt.close()

print("PCA visualization saved successfully!")
print("File: visualizations/01_pca_visualization.png")
# ========================================
# STEP 13 - ELBOW METHOD
# ========================================

from sklearn.cluster import KMeans

print("\n===== ELBOW METHOD =====")

inertia = []

# Test K values from 2 to 10
k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)

    inertia.append(kmeans.inertia_)

    print(
        f"K = {k}, "
        f"Inertia = {kmeans.inertia_:.2f}"
    )

# Plot Elbow Curve
plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/02_elbow_method.png",
    dpi=300
)

plt.close()

print("\nElbow Method graph saved successfully!")
print("File: visualizations/02_elbow_method.png")
# ========================================
# STEP 14 - SILHOUETTE SCORE
# ========================================

from sklearn.metrics import silhouette_score

print("\n===== SILHOUETTE SCORE =====")

silhouette_scores = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(scaled_data)

    score = silhouette_score(
        scaled_data,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.4f}"
    )

# Find best K
best_k = list(k_values)[
    np.argmax(silhouette_scores)
]

print("\nBest K based on Silhouette Score:", best_k)

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal K")

plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/03_silhouette_score.png",
    dpi=300
)

plt.close()

print("\nSilhouette Score graph saved successfully!")
print(
    "File: visualizations/03_silhouette_score.png"
)
# ========================================
# STEP 15 - FINAL K-MEANS CLUSTERING
# ========================================

print("\n===== FINAL K-MEANS CLUSTERING =====")

# Create final K-Means model
final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

# Create cluster labels
df["Cluster"] = final_kmeans.fit_predict(scaled_data)

print("Final K-Means clustering completed successfully!")

print("\nOptimal Number of Clusters:", best_k)

print("\nCluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

print("\nFirst 10 customers with cluster labels:")
print(
    df[
        ["Customer_ID", "Cluster"]
    ].head(10)
)

# Save clustered dataset
df.to_csv(
    "output/clustered_customer_data.csv",
    index=False
)

print("\nClustered dataset saved successfully!")
print("File: output/clustered_customer_data.csv")
# ========================================
# STEP 16 - CUSTOMER CLUSTER VISUALIZATION
# ========================================

print("\n===== CUSTOMER CLUSTER VISUALIZATION =====")

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=df["Cluster"],
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means and PCA")

plt.colorbar(
    scatter,
    label="Customer Cluster"
)

plt.tight_layout()

plt.savefig(
    "visualizations/04_customer_clusters.png",
    dpi=300
)

plt.close()

print("Customer cluster visualization saved successfully!")
print("File: visualizations/04_customer_clusters.png")
# ========================================
# STEP 17 - CLUSTER ANALYSIS
# ========================================

print("\n===== CLUSTER ANALYSIS =====")

# Calculate average values for each cluster
cluster_summary = df.groupby("Cluster").mean(
    numeric_only=True
)

print("\nAverage characteristics of each cluster:")
print(cluster_summary)

# Save cluster summary
cluster_summary.to_csv(
    "output/cluster_summary.csv"
)

print("\nCluster summary saved successfully!")
print("File: output/cluster_summary.csv")
# ========================================
# STEP 18 - CUSTOMER PERSONAS
# ========================================

print("\n===== CUSTOMER PERSONAS =====")

# Create persona names dynamically
persona_names = {}

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = cluster_summary.loc[cluster]

    income = cluster_data["Annual_Income"]
    spending = cluster_data["Spending_Score"]
    frequency = cluster_data["Purchase_Frequency"]

    # Determine customer persona
    if income >= cluster_summary["Annual_Income"].median() and spending >= cluster_summary["Spending_Score"].median():
        persona = "Premium High-Value Customers"

    elif income >= cluster_summary["Annual_Income"].median() and spending < cluster_summary["Spending_Score"].median():
        persona = "Affluent Low-Spending Customers"

    elif income < cluster_summary["Annual_Income"].median() and spending >= cluster_summary["Spending_Score"].median():
        persona = "Budget Enthusiastic Customers"

    else:
        persona = "Low-Engagement Customers"

    persona_names[cluster] = persona


# Add persona column
df["Customer_Persona"] = df["Cluster"].map(persona_names)

print("\nCustomer Persona Mapping:")

for cluster, persona in persona_names.items():
    print(f"Cluster {cluster} -> {persona}")


print("\nCustomer Persona Distribution:")
print(df["Customer_Persona"].value_counts())


# Save persona dataset
df.to_csv(
    "output/customer_personas.csv",
    index=False
)

print("\nCustomer persona dataset saved successfully!")
print("File: output/customer_personas.csv")
# ========================================
# STEP 19 - CUSTOMER PERSONA VISUALIZATION
# ========================================

print("\n===== CUSTOMER PERSONA VISUALIZATION =====")

persona_counts = df["Customer_Persona"].value_counts()

plt.figure(figsize=(10, 6))

plt.bar(
    persona_counts.index,
    persona_counts.values
)

plt.xlabel("Customer Persona")
plt.ylabel("Number of Customers")
plt.title("Customer Persona Distribution")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "visualizations/05_customer_personas.png",
    dpi=300
)

plt.close()

print("Customer persona visualization saved successfully!")
print("File: visualizations/05_customer_personas.png")
# ========================================
# STEP 20 - BUSINESS INSIGHTS & FINAL SUMMARY
# ========================================

print("\n===== BUSINESS INSIGHTS =====")

summary_file = "output/final_project_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("========================================\n")
    file.write("   CUSTOMER SEGMENTATION PROJECT\n")
    file.write("========================================\n\n")

    file.write("PROJECT: Unsupervised Learning - Customer Segmentation\n\n")

    file.write("1. DATASET SUMMARY\n")
    file.write("------------------\n")
    file.write(f"Total Customers: {len(df)}\n")
    file.write(f"Total Features Used: {features.shape[1]}\n\n")

    file.write("2. PCA SUMMARY\n")
    file.write("--------------\n")
    file.write(f"Original Dimensions: {scaled_data.shape[1]}\n")
    file.write(f"Reduced Dimensions: {pca_data.shape[1]}\n")
    file.write(
        f"Total Explained Variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}\n\n"
    )

    file.write("3. CLUSTERING SUMMARY\n")
    file.write("---------------------\n")
    file.write(f"Optimal Number of Clusters: {best_k}\n\n")

    file.write("4. CLUSTER DISTRIBUTION\n")
    file.write("-----------------------\n")

    for cluster, count in df["Cluster"].value_counts().sort_index().items():
        file.write(
            f"Cluster {cluster}: {count} customers\n"
        )

    file.write("\n5. CUSTOMER PERSONAS\n")
    file.write("--------------------\n")

    for cluster, persona in persona_names.items():
        count = (df["Cluster"] == cluster).sum()

        file.write(
            f"Cluster {cluster}: {persona} "
            f"({count} customers)\n"
        )

    file.write("\n6. BUSINESS INTERPRETATION\n")
    file.write("---------------------------\n")
    file.write(
        "Customer segmentation helps identify groups of "
        "customers with similar characteristics.\n"
    )
    file.write(
        "These segments can support targeted marketing, "
        "personalized offers and customer engagement strategies.\n"
    )

    file.write("\n7. CONCLUSION\n")
    file.write("-------------\n")
    file.write(
        "PCA was used for dimensionality reduction and visualization. "
        "K-Means clustering was applied to identify customer groups. "
        "The Elbow Method and Silhouette Score were used to evaluate "
        "the appropriate number of clusters. The resulting clusters "
        "were translated into customer personas for business insights.\n"
    )

print("Business insights generated successfully!")
print(f"Final summary saved to: {summary_file}")
