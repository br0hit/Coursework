

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import Naive Bayes, SVM, Decision Tree, and KNN from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

# Import metrics to evaluate the model and also the cross validation
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from math import sqrt, pi, exp


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from sklearn.impute import SimpleImputer

# Calculate prior probabilities
def calculate_prior_probabilities(y_train):
    class_labels, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    return {class_value: count / total_samples for class_value, count in zip(class_labels, class_counts)}

# Calculate Gaussian Probability Density
def gaussian_probability(x, mean, var):
    if var == 0:  # To avoid division by zero
        var = 1e-4
    exponent = exp(-((x - mean)**2 / (2 * var)))
    return (1 / sqrt(2 * pi * var)) * exponent

# Calculate the class probabilities for a given input sample
def calculate_class_probabilities(summaries, input_data, priors):
    probabilities = {}

    for class_value, class_summary in summaries.items():
        probabilities[class_value] = priors[class_value]  # Start with the prior probability

        for feature, value in input_data.items():
            mean, var = class_summary[feature]
            probabilities[class_value] *= gaussian_probability(value, mean, var)

    return probabilities


# Predict class for a single data point
def predict(summaries, input_data, priors):
    probabilities = calculate_class_probabilities(summaries, input_data, priors)
    # Return the class with the highest probability
    return max(probabilities, key=probabilities.get)

# Predict for the entire test set
def predict_all(summaries, X_test, priors):
    predictions = []
    for _, row in X_test.iterrows():
        result = predict(summaries, row, priors)
        predictions.append(result)
    return np.array(predictions)

# Accuracy calculation
def accuracy(y_true, y_pred):
    correct = sum(y_true == y_pred)
    return correct / len(y_true)

# Calculate the mean and variance for each feature by class
def calculate_mean_variance_by_class(X_train, y_train):
    summaries = defaultdict(dict)

    # Separate the data by class (0: Innocent, 1: Criminal)
    class_labels = np.unique(y_train)
    for class_value in class_labels:
        X_class = X_train[y_train == class_value]
        summaries[class_value] = {
            col: (X_class[col].mean(), X_class[col].var()) for col in X_train.columns
        }
    return summaries

# Function to compute precision, recall, and F1 score
def precision_recall_f1(y_true, y_pred):
    # Convert inputs to numpy arrays for element-wise comparison
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Positives (TP): Correctly predicted positive instances
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # False Positives (FP): Negative instances predicted as positive
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # False Negatives (FN): Positive instances predicted as negative
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

df = pd.read_csv('Data.csv')

# Using the new columns, we can now split the data into training and testing sets
# Define the features and the target variables

# First use the original df to split the data
original_df = pd.read_csv('Data.csv')

# Drop the rows with missing values
original_df = original_df.dropna()

# Covnert the float64 columns to int64 in 'hoursperweek'
original_df['hoursperweek'] = original_df['hoursperweek'].astype('int64')

# Create capitalloss_binary and capitalgain_binary columns by assigning 1 if the value is greater than 0, otherwise 0
original_df['capitalloss_binary'] = original_df['capitalloss'].apply(lambda x: 1 if x > 0 else 0)
original_df['capitalgain_binary'] = original_df['capitalgain'].apply(lambda x: 1 if x > 0 else 0)

# Only covert the 'Possibility' column to binary values
original_df['Possibility'] = original_df['Possibility'].map({'<=0.5': 1, '>0.5': 0})

# Perform binary encoding on the 'sex' column
original_df['sex'] = original_df['sex'].map({'Male':1, 'Female':0})

# Perform label encoding on the 'education' column
original_df['education_label'] = original_df['education'].map({
    'Preschool': 0,
    '1st-4th': 1,
    '5th-6th': 1,
    '7th-8th': 1,
    '9th':1,
    '10th': 2,
    '11th': 2,
    '12th': 2,
    'HS-grad': 3,
    'Some-college': 4,
    'Assoc-acdm': 5,
    'Assoc-voc': 5,
    'Bachelors': 6,
    'Masters': 7,
    'Doctorate': 8,
    'Prof-school': 8})

original_df['native_label'] = original_df['native'].map({
    # 0-5% range
    'Dominican-Republic': 0,
    'Outlying-US(Guam-USVI-etc)': 0,
    'Columbia': 0,
    'Guatemala': 0,

    # 5-10% range
    'Mexico': 1,

    'Nicaragua': 1,
    'Peru': 1,
    'Vietnam': 1,
    'Honduras': 1,
    'El-Salvador': 1,
    'Haiti': 1,


    # 10-15% range
    'Puerto-Rico': 2,
    'Trinadad&Tobago': 2,
    'Portugal': 2,
    'Laos': 2,
    'Ecuador': 2,
    'Jamaica': 2,


    # 15-20% range
    'Thailand': 3,
    'Ireland': 3,
    'South': 3,
    'Scotland': 3,
    'Poland': 3,

    # 20-25% range
    'Hungary': 4,
    'United-States': 4,

    # 25-30% range
    'Cuba': 5,
    'China': 5,
    'Greece': 5,


    # 30-35% range
    'Philippines': 6,
    'Hong': 6,
    'Canada': 6,
    'Germany': 6,
    'England': 6,
    'Italy': 6,

    # 35-40% range
    'Yugoslavia': 7,
    'Cambodia': 7,
    'Japan': 7,

    # 40-45% range
    'India': 8,
    'Iran': 8,
    'France': 8,
    'Taiwan': 8
})

original_df['native_onehot'] = original_df['native'].map({
    # 0-5% range
    'Dominican-Republic': 0,
    'Outlying-US(Guam-USVI-etc)': 0,
    'Columbia': 0,
    'Guatemala': 0,

    # 5-10% range
    'Mexico': 1,

    'Nicaragua': 1,
    'Peru': 1,
    'Vietnam': 1,
    'Honduras': 1,
    'El-Salvador': 1,
    'Haiti': 1,


    # 10-15% range
    'Puerto-Rico': 2,
    'Trinadad&Tobago': 2,
    'Portugal': 2,
    'Laos': 2,
    'Ecuador': 2,
    'Jamaica': 2,


    # 15-20% range
    'Thailand': 3,
    'Ireland': 3,
    'South': 3,
    'Scotland': 3,
    'Poland': 3,

    # 20-25% range
    'Hungary': 4,
    'United-States': 4,

    # 25-30% range
    'Cuba': 5,
    'China': 5,
    'Greece': 5,


    # 30-35% range
    'Philippines': 6,
    'Hong': 6,
    'Canada': 6,
    'Germany': 6,
    'England': 6,
    'Italy': 6,

    # 35-40% range
    'Yugoslavia': 'Yugoslavia',
    'Cambodia': 'Cambodia',
    'Japan':  'Japan',

    # 40-45% range
    'India': 'India',
    'Iran': 'Iran',
    'France': 'France',
    'Taiwan': 'Taiwan'
})

# Perform binary encoding on the 'maritalstatus' column
original_df['maritalstatus_label'] = original_df['maritalstatus'].map({
    'Never-married': 0,
    'Divorced': 0,
    'Separated': 0,
    'Widowed': 0,
    'Married-spouse-absent': 0,
    'Married-civ-spouse': 1,
    'Married-AF-spouse': 1})


# Define custom categories based on the analysis
original_df['maritalstatus_onehot'] = original_df['maritalstatus'].map({
    'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married',
    'Divorced': 'Divorced-Widowed-Abs',
    'Widowed': 'Divorced-Widowed-Abs',
    'Married-spouse-absent': 'Divorced-Widowed-Abs',
    'Separated': 'Separated-Never-Married',
    'Never-married': 'Separated-Never-Married'
})

# Define custom categories for the relationship column
original_df['relationship_onehot'] = original_df['relationship'].map({
    'Wife': 'inrelation',
    'Husband': 'inrelation',
    'Not-in-family': 'Not-in-family',
    'Unmarried': 'Unmarried',
    'Other-relative': 'Other-relative',
    'Own-child': 'Own-child'
    })

# Define custom categories for the occupation column
original_df['occupation_onehot'] = original_df['occupation'].map({
    'Exec-managerial': 'Exec-prof',
    'Prof-specialty': 'Exec-prof',
    'Protective-serv': 'Protective-Tech-Sales',
    'Tech-support': 'Protective-Tech-Sales',
    'Sales': 'Protective-Tech-Sales',
    'Craft-repair': 'Craft-Transp',
    'Transport-moving': 'Craft-Transp',
    'Adm-clerical': 'Admin-Machine-farm-armed',
    'Machine-op-inspct': 'Admin-Machine-farm-armed',
    'Farming-fishing': 'Admin-Machine-farm-armed',
    'Handlers-cleaners': 'cleaners-others',
    'Other-service': 'cleaners-others',
    'Priv-house-serv': 'Priv-house-serv',
})

print(original_df.info())

# Use the following features to train the model
features_NaiveBayes_numeric = ['age', 'educationno', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'Possibility']
features_NaiveBayes_categorical = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'native']

# Use the following features to train the model
features_NaiveBayes = features_NaiveBayes_numeric + features_NaiveBayes_categorical

NaiveBayes_df = original_df[features_NaiveBayes]

# Conver the categorical columns to one-hot encoding
NaiveBayes_df = pd.get_dummies(NaiveBayes_df, columns=features_NaiveBayes_categorical, drop_first=True)

# Define the features and the target variables
X = NaiveBayes_df.drop(['Possibility'], axis=1)
y = NaiveBayes_df['Possibility']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the training sets
# Naive Bayes

# Create a Gaussian Classifier
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_NaiveBayes = gnb.predict(X_test)

# Find the model accuracy and the F1 score
print("Naive Bayes")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_NaiveBayes))
print("F1 Score:", metrics.f1_score(y_test, y_pred_NaiveBayes))

feautures_NaiveBayes_custom_numeric = ['age', 'educationno', 'sex', 'capitalgain', 'capitalloss_binary', 'hoursperweek', 'Possibility']
features_NaiveBayes_custom_categorical = ['workclass', 'education_label', 'maritalstatus', 'occupation_onehot', 'relationship', 'race', 'native_onehot']
features_NaiveBayes_custom = feautures_NaiveBayes_custom_numeric + features_NaiveBayes_custom_categorical
NaiveBayes_custom_df = original_df[features_NaiveBayes_custom]
NaiveBayes_custom_df = pd.get_dummies(NaiveBayes_custom_df, columns=features_NaiveBayes_custom_categorical, drop_first=True)

X = NaiveBayes_custom_df.drop(['Possibility'], axis=1)
y = NaiveBayes_custom_df['Possibility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Summarize the training set
class_summaries = calculate_mean_variance_by_class(X_train, y_train)
priors = calculate_prior_probabilities(y_train)


# Get predictions for the test set
y_pred_NaiveBayes_custom = predict_all(class_summaries, X_test, priors)

# Evaluate accuracy on the test set
test_accuracy = accuracy(y_test, y_pred_NaiveBayes_custom)
print(f"Test Accuracy: {test_accuracy :.2f}")

# Calculate precision, recall, and F1 score on the test set
precision, recall, f1 = precision_recall_f1(y_test, y_pred_NaiveBayes_custom)

# Output the results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print(y_pred_NaiveBayes_custom)

feautures_custom_numeric = ['age', 'educationno', 'sex', 'capitalgain_binary', 'capitalloss_binary', 'hoursperweek', 'Possibility']
features_custom_categorical = ['workclass', 'education_label', 'maritalstatus', 'occupation_onehot', 'relationship', 'race', 'native_onehot']
features_custom = feautures_custom_numeric + features_custom_categorical
custom_df = original_df[features_custom]
custom_df = pd.get_dummies(custom_df, columns=features_custom_categorical, drop_first=True)

print(custom_df.info())


X = custom_df.drop(['Possibility'], axis=1)
y = custom_df['Possibility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM

# Create a SVM Classifier
clf = svm.SVC(kernel='linear') # Linear
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)

print("SVM")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
print("F1 Score:", metrics.f1_score(y_test, y_pred_svm))

# Decision Tree

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred_dt = clf.predict(X_test)

print("Decision Tree")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_dt))
print("F1 Score:", metrics.f1_score(y_test, y_pred_dt))

# KNN

# Create a KNN Classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred_knn = neigh.predict(X_test)

print("KNN")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))
print("F1 Score:", metrics.f1_score(y_test, y_pred_knn))

# Define Custom Naive Bayes Classifier
class CustomNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = X_cls.mean(axis=0)
            self.vars[cls] = X_cls.var(axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            probs = []
            for cls in self.classes:
                mean = self.means[cls]
                var = self.vars[cls]
                prior = self.priors[cls]

                prob = np.log(prior)
                prob -= 0.5 * np.sum(np.log(2 * np.pi * var))
                prob -= 0.5 * np.sum(((x - mean) ** 2) / var)

                probs.append(prob)

            predictions.append(self.classes[np.argmax(probs)])
        return np.array(predictions)

# Initialize classifiers
naive_bayes = CustomNaiveBayes()
svm_model = svm.SVC(kernel='linear')
decision_tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()

# Train the classifiers with imputed data
naive_bayes.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Define the Ensemble Classifier
class EnsembleClassifier:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        # Gather predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])

        # Use majority voting for the final prediction
        final_predictions = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)

        return final_predictions



# Initialize the ensemble with the classifiers
ensemble_model = EnsembleClassifier(models=[naive_bayes, svm_model, decision_tree, knn])

# Train the ensemble classifier
ensemble_model.fit(X_train, y_train)

# Predict and evaluate on the test data
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class classification

print(f"Ensemble Model Accuracy: {accuracy: }")
print(f"Ensemble Model F1 Score: {f1: }")