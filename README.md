# AI vs Human Text Detection

> NLP project to distinguish between AI-generated and human-written text using classical machine learning and a transformer model.

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Models](#models)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Team](#team)
- [Technologies Used](#technologies-used)

## Overview
With the rise of large language models like ChatGPT, distinguishing AI-generated text from human-written content has become a critical challenge. This project builds and evaluates binary classifiers that detect whether a given text was written by a human or an AI.

## Motivation
AI-generated text can be misused in:
- **Education** – students submitting AI-written essays
- **Journalism** – spreading fake news or uncredited AI articles
- **Online platforms** – bots and fake reviews

Our goal is to create a robust detection model to help mitigate these issues.

## Dataset
- **Source**: [Kaggle – AI vs Human Text Dataset](https://www.kaggle.com/) (exact link not provided)
- **Total samples**: 200,000
- **Human texts**: 125,349 (63%)
- **AI texts**: 74,651 (37%)
- **Labels**: `0` = human, `1` = AI
- **Average text length**: ~2,270 characters

## Preprocessing
All text data was cleaned using the following steps:

1. **Lowercasing** – convert all characters to lowercase
2. **Tokenization** – split text into individual tokens
3. **Remove punctuation & stopwords** – filter out noise
4. **Lemmatization** – reduce words to their base form

**Tools**: NLTK, Joblib (for parallel processing)

## Feature Extraction
- **Method**: TF-IDF (Term Frequency – Inverse Document Frequency)
- **N-gram range**: Unigrams + Bigrams
- **Vocabulary size**: Top 2,000 features

TF-IDF assigns higher weights to words that appear frequently in a document but rarely across the entire corpus. This helps capture distinguishing phrases like *"in conclusion"*, *"additionally"*, and *"however"* often found in AI-generated text.

## Models

### Classical Models (with TF-IDF features)
- Logistic Regression
- Naïve Bayes
- Linear SVM
- Random Forest
- Gradient Boosting

### Transformer Model
- **DistilRoBERTa** – a lightweight, faster version of RoBERTa, trained on 20k samples as a deep learning baseline.

## Experimental Setup
- **Data split**: 80% training / 20% testing
- **Classical models**: Trained on full TF-IDF vectors
- **Transformer**: Trained on 20k samples due to computational constraints
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score

## Results

| Model | F1-Score |
|-------|----------|
| Random Forest | **0.9951** |
| Other classical models | ~0.98–0.99 |
| DistilRoBERTa | ~0.98 |

Random Forest achieved the best performance across all metrics, with an F1-score of 0.9951.

## Limitations
Despite high accuracy on the benchmark dataset, the models have significant weaknesses:

- **Casual human text** – Slang, typos, and informal writing confuse the models.
- **Formal human writing** – Polished human text is sometimes misclassified as AI.
- **Dataset bias** – Human texts in the training set are predominantly informal, while AI texts are formal and structured. The models learned formality rather than true authorship signals.

## Conclusion
- We built effective binary classifiers that perform exceptionally well on the provided dataset (up to 99.6% accuracy).
- However, both classical and transformer models fail on real-world, casual human text due to dataset bias.
- **Key takeaway**: True progress in NLP requires robustness, not just leaderboard accuracy. Real-world evaluation is essential.

## Team
- Fatena
- Keşkül
- Ateh

**Course**: BIM432 – Natural Language Processing  
**Institution**: Istanbul Zaim University  
**Date**: May 2026

## Technologies Used
- Python
- NLTK
- Joblib
- Scikit-learn (TF-IDF, Logistic Regression, SVM, Random Forest, etc.)
- Transformers (Hugging Face) – DistilRoBERTa

## How to Run (Brief)
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle.
4. Run preprocessing script: `python preprocess.py`
5. Train classical models: `python train_classical.py`
6. Train DistilRoBERTa: `python train_transformer.py`
7. Evaluate: `python evaluate.py`

> *Note: This README is based on the project presentation. For full code, please refer to the repository files.*
