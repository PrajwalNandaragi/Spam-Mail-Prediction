# Spam Email Detection using Logistic Regression

## Overview
This project implements a spam email detection system using **Logistic Regression**. It utilizes **TF-IDF Vectorization** for text processing and classification, and achieves high accuracy in distinguishing between spam and ham (non-spam) emails.

## Dataset
The dataset used for this project is a collection of SMS messages labeled as either spam or ham. The dataset is read from a CSV file and preprocessed to handle missing values.

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn (sklearn)

## Steps Involved
### 1. Data Preprocessing
- Load the dataset from `mail_data.csv`
- Replace missing values with empty strings
- Encode the target labels: **spam = 0, ham = 1**
- Split the dataset into training and testing sets

### 2. Feature Extraction
- Convert text messages into numerical vectors using **TF-IDF Vectorization**
- Remove stop words to improve model performance

### 3. Model Training
- Train a **Logistic Regression Model** on the feature-extracted training data
- Evaluate performance using **accuracy score**

### 4. Model Evaluation
- Compute accuracy on training and testing datasets
- Achieve around **96.7% accuracy on training data** and **96.6% on test data**

### 5. Prediction
- Input new email messages for classification
- Convert input text into feature vectors
- Predict whether the email is **Spam (0)** or **Ham (1)**

## Example Prediction
Input:
```python
input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
```
Output:
```python
[0]
Spam mail
```

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-email-detector.git
   cd spam-email-detector
   ```
2. Install required dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Run the script:
   ```bash
   python spam_detector.py
   ```

## Conclusion
This project successfully detects spam emails using **TF-IDF vectorization** and **Logistic Regression**, achieving high accuracy. Future improvements could involve deep learning techniques or additional NLP preprocessing for better results.

## License
This project is licensed under the MIT License.

