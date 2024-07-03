# SentimentAnalysis

This project aims to build sentiment classifiers using bag of words and n-gram models for a dataset containing reviews. The reviews are labeled as 1 (negative), 3 (neutral), 5 (positive), or not_relevant. The goal is to perform multi-class classification and accurately categorize the reviews into one of the four classes.

### Dataset

The dataset consists of reviews with the following labels:
- 1 (negative)
- 3 (neutral) 
- 5 (positive)
- not_relevant

The dataset is provided in the attached file: [`dataset_sentiment.csv`](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/23568731/ab9b56c6-5915-45a2-a8a8-8b4216a7018d/dataset_sentiment.csv).

### Methodology

1. **Data Preprocessing**:
   - The dataset is split into training and test sets using `train_test_split()` from scikit-learn.

2. **Feature Extraction**:
   - Bag of words based on raw counts
   - Bag of words based on TF-IDF
   - N-grams (unigrams, bigrams, trigrams)

3. **Classifiers**:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - SVM
   - Perceptron

4. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F-score
   - Micro average and macro average for all measures

### Results

The results are presented in tables for multi-class classification, comparing the performance of different classification algorithms. The tables include accuracy, precision, recall, and F-score for each classifier.

### Code

The code is implemented in Python using scikit-learn and is provided in the attached Jupyter Notebook file: [`SentimentAnalysis.ipynb`](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/23568731/0714e5e0-8f60-4d7c-9187-c2b4d0c3b1ef/SentimentAnalysis.ipynb).

### Usage

1. Ensure you have Python and the necessary libraries (scikit-learn) installed.
2. Open the `SentimentAnalysis.ipynb` file in a Jupyter Notebook environment.
3. Run the code cells to preprocess the data, extract features, train the classifiers, and evaluate their performance.
4. Analyze the results and compare the performance of different classifiers.

### Conclusion

This project demonstrates the implementation of sentiment analysis on a dataset of Apple product reviews using various classification algorithms and feature extraction techniques. The results provide insights into the performance of different classifiers and can be used to select the most suitable approach for sentiment analysis tasks.
