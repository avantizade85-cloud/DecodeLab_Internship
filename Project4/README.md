DecodeLab Data Science Internship – Project 4
NLP & Sentiment Analysis
📌 Project Overview

This project is part of the DecodeLab Data Science Internship and focuses on Natural Language Processing (NLP) and Sentiment Analysis.

The main objective of this project is to analyze unstructured human text, such as product reviews, and classify the reviews into Positive or Negative sentiment categories.

The project follows a complete NLP pipeline including text preprocessing, TF-IDF vectorization, machine learning model training, evaluation, and prediction of new reviews.

🎯 Objectives
Process unstructured product review text.
Perform NLP text preprocessing.
Tokenize the review text.
Remove stop words.
Apply lemmatization.
Convert text into numerical features using TF-IDF.
Train a machine learning classification model.
Evaluate model performance.
Predict sentiment for new reviews.
📊 Dataset

The project uses a product review dataset containing 100 reviews.

Dataset Details
Feature	Description
review	Product review text
sentiment	Positive or Negative sentiment
Dataset Distribution
Total Reviews: 100
Positive Reviews: 50
Negative Reviews: 50

The original dataset is stored in:

dataset/reviews_dataset.csv
🛠️ Technologies Used
Python
Pandas
NLTK
Scikit-learn
Matplotlib
🔄 NLP Preprocessing

The following preprocessing steps were performed on the review text:

1. Lowercasing

All review text was converted into lowercase.

2. Removing Special Characters

Punctuation and unnecessary special characters were removed.

3. Tokenization

The review text was split into individual words using NLTK tokenization.

4. Stop-word Removal

Common English stop words were removed using NLTK.

5. Lemmatization

Words were converted into their base form using WordNetLemmatizer.

Example

Original Review:

This product is amazing and I really love it

After Preprocessing:

product amazing really love

The cleaned dataset is saved as:

dataset/cleaned_reviews.csv
🔢 TF-IDF Vectorization

After preprocessing the text, TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the reviews into numerical feature vectors.

The resulting TF-IDF matrix contains:

100 samples × 131 features

This numerical representation is then used as input for the machine learning model.

🤖 Machine Learning Model

A Multinomial Naive Bayes classifier was used for sentiment classification.

Train-Test Split

The dataset was divided into:

80% Training Data
20% Testing Data

The model was trained using the TF-IDF feature matrix.

📈 Model Evaluation

The trained model was evaluated using the following metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

The detailed model results are stored in:

output/model_results.txt
📊 Visualizations

The project generates the following visualizations.

1. Confusion Matrix

The confusion matrix shows the actual and predicted sentiment classifications.

File:

visualizations/confusion_matrix.png
2. Sentiment Distribution

This graph shows the distribution of Positive and Negative reviews in the dataset.

File:

visualizations/sentiment_distribution.png
🔮 New Review Prediction

The trained model can predict the sentiment of a new product review.

Example 1
Review:
This product is amazing and I really love it

Prediction:
Positive
Example 2
Review:
The product is very bad and I hate it

Prediction:
Negative
📂 Project Structure
DecodeLab_DataScience_Project4/
│
├── Main.py
├── README.md
│
├── dataset/
│   ├── reviews_dataset.csv
│   └── cleaned_reviews.csv
│
├── output/
│   └── model_results.txt
│
└── visualizations/
    ├── confusion_matrix.png
    └── sentiment_distribution.png
▶️ How to Run the Project
Step 1: Clone the Repository
git clone <repository-url>
Step 2: Open the Project

Open the project folder in VS Code.

Step 3: Install Required Libraries
pip install pandas nltk scikit-learn matplotlib
Step 4: Run the Project
python Main.py

For the Python installation used during development:

& "C:\Program Files\Python314\python.exe" "Main.py"
📁 Generated Files

After running the project, the following files are generated:

dataset/cleaned_reviews.csv
output/model_results.txt
visualizations/confusion_matrix.png
visualizations/sentiment_distribution.png
💡 Key Learnings

Through this project, the following concepts were implemented:

Natural Language Processing
Text Cleaning
Tokenization
Stop-word Removal
Lemmatization
TF-IDF Vectorization
Text Classification
Naive Bayes
Model Evaluation
Sentiment Prediction
Data Visualization


✅ Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be used to classify unstructured product reviews into Positive and Negative sentiments.

The complete workflow includes text preprocessing, tokenization, stop-word removal, lemmatization, TF-IDF vectorization, Naive Bayes model training, performance evaluation, visualization, and prediction of new reviews.

The project successfully implements an end-to-end NLP Sentiment Analysis pipeline as part of the DecodeLab Data Science Internship – Project 4.