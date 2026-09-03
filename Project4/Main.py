import pandas as pd
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# STEP 1: Download NLTK Resources
# ==========================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ==========================================
# STEP 2: Load Dataset
# ==========================================

df = pd.read_csv("dataset/reviews_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())


# ==========================================
# STEP 3: Initialize NLP Tools
# ==========================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ==========================================
# STEP 4: Text Preprocessing Function
# ==========================================

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stop-word removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Convert tokens back to text
    return " ".join(tokens)


# ==========================================
# STEP 5: Apply Text Preprocessing
# ==========================================

df["cleaned_review"] = df["review"].apply(preprocess_text)

print("\nOriginal vs Cleaned Text:")
print(df[["review", "cleaned_review"]].head(10))


# ==========================================
# STEP 6: Save Cleaned Dataset
# ==========================================

df.to_csv("dataset/cleaned_reviews.csv", index=False)

print("\nCleaned dataset saved successfully!")


# ==========================================
# STEP 7: TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["cleaned_review"])

print("\nTF-IDF Vectorization completed successfully!")

print("TF-IDF Matrix Shape:", X.shape)

print(
    "Number of Features:",
    len(vectorizer.get_feature_names_out())
)

print("\nFirst 10 Features:")
print(vectorizer.get_feature_names_out()[:10])
# ==========================================
# STEP 7: Train-Test Split
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Convert sentiment labels into numbers
y = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# STEP 8: Train Naive Bayes Model
# ==========================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nNaive Bayes model trained successfully!")


# ==========================================
# STEP 9: Model Prediction
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy Percentage:", round(accuracy * 100, 2), "%")


# ==========================================
# STEP 10: Classification Report
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)


# ==========================================
# STEP 11: Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# ==========================================
# STEP 12: Confusion Matrix Visualization
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Values inside matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.tight_layout()

plt.savefig("visualizations/confusion_matrix.png")

plt.show()

print("\nConfusion Matrix graph saved successfully!")
# ==========================================
# STEP 13: Sentiment Distribution
# ==========================================

plt.figure(figsize=(6, 5))

df["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("visualizations/sentiment_distribution.png")

plt.show()

print("Sentiment distribution graph saved successfully!")
# ==========================================
# STEP 14: New Review Prediction
# ==========================================

def predict_sentiment(review):

    # Preprocess new review
    review_cleaned = preprocess_text(review)

    # Convert review into TF-IDF
    review_vector = vectorizer.transform([review_cleaned])

    # Predict sentiment
    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


# Test new reviews

new_review_1 = "This product is amazing and I really love it"

new_review_2 = "The product is very bad and I hate it"

print("\nNew Review Predictions:")

print("Review:", new_review_1)
print("Prediction:", predict_sentiment(new_review_1))

print("\nReview:", new_review_2)
print("Prediction:", predict_sentiment(new_review_2))
# ==========================================
# STEP 15: Save Model Results
# ==========================================

import os

os.makedirs("output", exist_ok=True)

with open("output/model_results.txt", "w") as f:

    f.write("NLP & SENTIMENT ANALYSIS - PROJECT 4\n")
    f.write("=" * 50 + "\n\n")

    f.write("Dataset Shape: " + str(df.shape) + "\n")
    f.write("Number of TF-IDF Features: " + str(X.shape[1]) + "\n\n")

    f.write("Training Samples: " + str(X_train.shape[0]) + "\n")
    f.write("Testing Samples: " + str(X_test.shape[0]) + "\n\n")

    f.write("Model: Multinomial Naive Bayes\n")
    f.write("Accuracy: " + str(round(accuracy * 100, 2)) + "%\n\n")

    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Positive"]
        )
    )

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nModel results saved successfully!")