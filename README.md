Spam Mail Prediction Using Logistic Regression
Overview
This project demonstrates how to classify email messages as "Spam" or "Ham" (non-spam) using a machine learning model. We use Logistic Regression, a popular supervised learning algorithm, for this task. The dataset is a collection of emails with labeled categories: "ham" (non-spam) and "spam." We preprocess the text data, extract features using TF-IDF, and train the model on the processed data.

Objective
The goal of this project is to build a machine learning model that can predict whether an email is spam or not based on the message content. The project involves the following steps:

Data loading and preprocessing.

Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency).

Model training using Logistic Regression.

Model evaluation and prediction.

Prerequisites
Before running the code, ensure you have the following libraries installed:

numpy

pandas

scikit-learn

You can install the required dependencies using pip:

bash
Copy
pip install numpy pandas scikit-learn
Dataset
The dataset used in this project is a CSV file named mail_data.csv. It contains two columns:

Category: The label, either "ham" or "spam".

Message: The actual email message content.

Sample data:

Category	Message
ham	Go until jurong point, crazy.. Available only ...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tickets...
ham	U dun say so early hor... U c already then say...
ham	Nah I don't think he goes to usf, he lives around the corner...
Workflow
1. Data Loading & Preprocessing
We load the dataset from a CSV file and replace null values with empty strings.

python
Copy
raw_mail_data = pd.read_csv('/content/mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
2. Label Encoding
We convert the Category column into numerical values:

Spam = 0

Ham = 1

python
Copy
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
3. Feature Extraction
We extract features from the Message column using TF-IDF Vectorization, which converts text data into numerical vectors suitable for machine learning models.

python
Copy
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
4. Model Training
We use a Logistic Regression classifier to train the model.

python
Copy
model = LogisticRegression()
model.fit(X_train_features, Y_train)
5. Model Evaluation
After training the model, we evaluate its accuracy on both the training and test data.

python
Copy
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
6. Prediction on New Data
We also predict whether a given email is spam or ham using the trained model. Here's an example of making a prediction on a new email:

python
Copy
input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
7. Accuracy
Accuracy on Training Data: ~96.77%

Accuracy on Test Data: ~96.68%

Results
After training the Logistic Regression model, we achieve high accuracy in classifying emails. The model correctly identifies spam and ham emails based on their content.

Example of prediction:

Input Mail: "SIX chances to win CASH! From 100 to 20,000 pounds..."

Prediction: Spam mail

How to Run the Code
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/Spam-Mail-Prediction.git
Navigate to the project folder:

bash
Copy
cd Spam-Mail-Prediction
Ensure the dataset mail_data.csv is in the correct path or update the code to match the location.

Run the code:

bash
Copy
python spam_mail_classifier.py
Conclusion
This project demonstrates how to effectively classify email messages as spam or ham using machine learning techniques. Logistic Regression, coupled with TF-IDF for feature extraction, provides an efficient solution for this classification problem.

License
This project is licensed under the MIT License.
