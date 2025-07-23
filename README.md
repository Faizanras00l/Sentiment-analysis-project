# Sentiment Analysis Project

Unlock the magic of Natural Language Processing with this end-to-end Sentiment Analysis solution for Amazon Fine Food Reviews. This repository empowers you to train, evaluate, and deploy a robust sentiment classifier using state-of-the-art machine learning techniques.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Download NLTK Data](#3-download-nltk-data)
  - [4. Prepare the Dataset](#4-prepare-the-dataset)
  - [5. Using Pretrained Models](#5-using-pretrained-models)
  - [6. Run the Notebook](#6-run-the-notebook)
- [Custom Sentiment Prediction](#custom-sentiment-prediction)
- [Results & Visualization](#results--visualization)
- [License](#license)

---

## ✨ Overview

This project leverages machine learning and NLP to classify review sentiments as **positive**, **neutral**, or **negative**. It includes everything from data preprocessing to model training, evaluation, and visualization, with ready-to-use pretrained weights for instant predictions.

## 🚀 Features

- Automated data cleaning and preprocessing
- TF-IDF vectorization for feature extraction
- Logistic Regression classifier
- Detailed metrics and visualizations
- Pretrained model weights for fast deployment
- Custom review sentiment prediction

## 🗂️ Project Structure

```
Sentiment_analysis_project.ipynb
WeightsMatrix/
    sentiment_model.pkl
    tfidf_vectorizer.pkl
    test_data_and_predictions.pkl
Parameters/
    full_metrics_report.txt
    classification_report.csv
    confusion_matrix.png
    class_scores_barplot.png
    class_support_barplot.png
    sentiment_distribution.png
```

---

## 🛠️ Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project
```

### 2. Install Dependencies

Install all required Python packages in one magical command:

```sh
pip install nltk scikit-learn pandas matplotlib seaborn wordcloud joblib
```

### 3. Download NLTK Data

Ensure all necessary NLTK corpora are available:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Prepare the Dataset

Download the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle and place `Reviews.csv` in the project root directory.

### 5. Using Pretrained Models

Pretrained weights are stored in the `WeightsMatrix/` folder for instant predictions:

```python
import joblib

model = joblib.load('WeightsMatrix/sentiment_model.pkl')
vectorizer = joblib.load('WeightsMatrix/tfidf_vectorizer.pkl')
```

### 6. Run the Notebook

Open `Sentiment_analysis_project.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially to experience the full workflow.

---

## 🧙 Custom Sentiment Prediction

Predict the sentiment of any review in seconds:

```python
def preprocess(text):
    # ...same as in notebook...
    return cleaned_text

review = "Absolutely loved the taste and quality!"
cleaned = preprocess(review)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]
print(f"Sentiment: {prediction}")
```

---

## 📊 Results & Visualization

All evaluation metrics and insightful visualizations are automatically saved in the `Parameters/` folder for your analysis and reporting needs.

---

## 📄 License

This project is licensed under the MIT License.

---

**Ready to turn text into insights? Clone, run, and let the magic of