## Executive Summary

This project focuses on **sentiment analysis** using a **Stochastic Gradient Descent (SGD)** classifier and **Multi-Layer Perceptron (MLP)** model to predict emotions such as **sad, jealous, joyful,** and **terrified** based on textual data. The goal is to classify a given text (utterance) into one of these four sentiments. The data transformation and processing pipeline involves various natural language processing (NLP) techniques, along with model training, evaluation, and error analysis.

### **Steps Followed**:

1. **Data Transformation**:
   - **Dataset**:
     - **Train**: 97,968 records, **Test**: 720 records.
   - We retained only two columns: `context` and `utterance`, and filtered the data to include only the four selected sentiments (`sad`, `jealous`, `joyful`, and `terrified`).

2. **Data Preprocessing**:
   - Removed **punctuation**, converted text to **lowercase**, and applied **lemmatization**.
   - Created a **Bag of Words** (BOW) model using **CountVectorizer** which initially resulted in high dimensionality.
   - Addressed dimensionality by removing **stop words** and performed **TF-IDF vectorization** to reduce irrelevant features.

3. **Modeling and Evaluation**:
   - Used a **Stochastic Gradient Descent (SGD) classifier** to train the model on the TF-IDF feature set.
   - Achieved a **test accuracy** of **63.6%** and an **F1 score** of **0.637**.
   - Analysis using a **confusion matrix** showed that the classification of the "terrified" sentiment had the highest accuracy, while "jealous" had the lowest.

4. **Error Analysis**:
   - Conducted an analysis of misclassified examples and found that **context** played a crucial role in prediction errors. Words like "worried" in sentences labeled as "sad" were often misclassified as "terrified," demonstrating the model's difficulty in understanding nuanced contexts.

5. **Improvement Using Word2Vec**:
   - Integrated **Word2Vec embeddings** to better capture context and semantics, aiming to improve predictions.
   - Trained a **Multi-Layer Perceptron (MLP) classifier** on the Word2Vec transformed data, achieving a slightly better accuracy of **64.17%** and an F1 score of **0.642**.

6. **Challenges**:
   - The model struggles with capturing the proper **context** from text, leading to misclassification of similar sentiments.
   - **Word2Vec** embeddings partially helped address this, but further exploration of context-aware methods like **BERT** or other transformer-based models could yield better results.

---

This project highlights the importance of text preprocessing, dimensionality reduction, and the choice of appropriate embeddings in sentiment analysis tasks. Future work may involve leveraging more context-sensitive models like **transformers** to improve performance.
