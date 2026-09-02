\# Customer Segmentation using Unsupervised Learning



\## DecodeLab Data Science Internship – Project 3



\### Project Overview



This project focuses on customer segmentation using \*\*Unsupervised Machine Learning\*\* techniques.



The objective is to discover hidden customer groups from retail customer data without using predefined labels.



The project uses \*\*PCA, K-Means Clustering, Elbow Method, and Silhouette Score\*\* to identify and analyze customer segments.



\---



\## Objectives



\* Analyze customer behavior and purchasing characteristics.

\* Clean and prepare the customer dataset.

\* Standardize numerical features.

\* Apply PCA for dimensionality reduction.

\* Determine the optimal number of clusters.

\* Apply K-Means clustering.

\* Visualize customer clusters.

\* Create meaningful customer personas.

\* Generate actionable business insights.



\---



\## Technologies Used



\* Python

\* Pandas

\* NumPy

\* Matplotlib

\* Scikit-learn



\---



\## Machine Learning Techniques



\### 1. Data Preprocessing



Missing numerical values are treated using median imputation and duplicate records are removed.



\### 2. Feature Standardization



`StandardScaler` is used to standardize the numerical features before applying PCA and clustering.



\### 3. Principal Component Analysis (PCA)



PCA is used to reduce the high-dimensional customer dataset to two dimensions for visualization.



\### 4. Elbow Method



The Elbow Method is used to evaluate different values of K using K-Means inertia.



\### 5. Silhouette Score



Silhouette Score is used to mathematically evaluate cluster quality and identify the optimal number of clusters.



\### 6. K-Means Clustering



K-Means is applied using the selected optimal number of clusters to group customers with similar characteristics.



\---



\## Customer Personas



The resulting clusters are translated into business-oriented customer personas such as:



\* Premium High-Value Customers

\* Affluent Low-Spending Customers

\* Budget Enthusiastic Customers

\* Low-Engagement Customers



These personas can help businesses understand customer behavior and design targeted strategies.



\---



\## Project Structure



```text

Project-3

│

├── customer\_segmentation.py

├── dataset.csv

│

├── output

│   ├── clustered\_customer\_data.csv

│   ├── cluster\_summary.csv

│   ├── customer\_personas.csv

│   └── final\_project\_summary.txt

│

├── visualizations

│   ├── 01\_pca\_visualization.png

│   ├── 02\_elbow\_method.png

│   ├── 03\_silhouette\_score.png

│   ├── 04\_customer\_clusters.png

│   └── 05\_customer\_personas.png

│

└── README.md

```



\---



\## Visualizations



The project generates the following visualizations:



1\. PCA Visualization

2\. Elbow Method Curve

3\. Silhouette Score Graph

4\. Customer Cluster Visualization

5\. Customer Persona Distribution



\---



\## Business Insights



Customer segmentation can help businesses:



\* Identify high-value customers.

\* Understand different purchasing behaviors.

\* Create personalized marketing campaigns.

\* Provide targeted offers and discounts.

\* Improve customer engagement.

\* Develop customer-specific strategies.



\---



\## How to Run the Project



\### Step 1 – Install Required Libraries



```bash

pip install pandas numpy matplotlib scikit-learn

```



\### Step 2 – Run the Python Script



```bash

python customer\_segmentation.py

```



\### Step 3 – Check Outputs



Generated datasets will be available in the `output` folder.



Generated graphs will be available in the `visualizations` folder.



\---



\## Conclusion



This project demonstrates how \*\*Unsupervised Learning\*\* can be used for customer segmentation.



PCA was used for dimensionality reduction and visualization, while the Elbow Method and Silhouette Score helped evaluate the appropriate number of K-Means clusters.



The resulting customer clusters were converted into meaningful customer personas, providing useful business insights for targeted marketing and customer engagement.



\---



\## Internship



\*\*Program:\*\* DecodeLab Data Science Internship



\*\*Project:\*\* Project 3 – Unsupervised Learning (Customer Segmentation)



