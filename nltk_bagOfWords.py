# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:46:00 2023
@author: tobias.gnos
"""

#%% Large Language Model Artificial Textclassification
# This approach uses a LLM for text encoding and learns a downstream task for detection of artificial generated texts based on this encoding.
# import dataprocessing.dataset as ds
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

is_submission = False

#%% Define hardware usage.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device {device}')
#%% Functions
def load_dataset():

    train_set_1 = pd.read_csv("./data/train_drcat_04.csv")
    #using only data with label == 1
    #train_set_1 = train_set_1[train_set_1["label"]==1]
    train_set_1 = train_set_1[["text","label"]]
    train_set_1['text'] = train_set_1['text'].str.replace('\n', '')
    
    train_set_2 = pd.read_csv("./data/daigt_external_dataset.csv", sep=',')
    train_set_2 = train_set_2.rename(columns={'generated': 'label'})
    train_set_2 = train_set_2[["source_text"]]
    train_set_2.columns = ["text"]
    train_set_2['text'] = train_set_2['text'].str.replace('\n', '')
    train_set_2["label"] = 1

    train_set_3 = pd.read_csv("./data/train_essays_RDizzl3_seven_v1.csv")

    train_set = pd.concat([train_set_1,train_set_2,train_set_3])
    
    train_set.rename(columns={'label': 'generated',
                              'essay_id': 'id'}, inplace=True)
    
    X_train, X_val, y_train, y_val = train_test_split(train_set["text"],train_set["generated"],test_size=0.2)

    data_train = []
    data_val = []
    max_sequence_length = 0

    for ii in range(len(X_train)):
        data_train.append({'text': X_train.values[ii], 'generated': y_train.values[ii]})
        if len(X_train.values[ii]) > max_sequence_length: max_sequence_length=len(X_train.values[ii])
    for ii in range(len(X_val)):
        data_val.append({'text': X_val.values[ii], 'generated': y_val.values[ii]})
        if len(X_val.values[ii]) > max_sequence_length: max_sequence_length=len(X_val.values[ii])

    print(f'Number of Training Data: {len(y_train)}, Number of Validation Data: {len(y_val)}')

    return data_train, data_val, max_sequence_length

def plotConfusionMatrix(labels_val, labels_pred, Title = ''):
    conf_mat = confusion_matrix(labels_val, labels_pred)
    # Plot the confusion matrix with numbers
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, cmap=plt.cm.Blues)
    
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, f'{conf_mat[i, j]}', ha='center', va='center', color='red')
    
    plt.title(Title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.colorbar()
    plt.xticks(ticks=[0, 1], labels=['False', 'True'])
    plt.yticks(ticks=[0, 1], labels=['False', 'True'])
    plt.show()
    
#%%  Load & Preprocess Dataset
data_train, test_data, max_sequence_length = load_dataset()

#%%  Extract texts and labels
texts_train = [item["text"] for item in data_train]
labels_train = [item["generated"] for item in data_train]
texts_val = [item["text"] for item in test_data]
labels_val = [item["generated"] for item in test_data]

# Tokenization using NLTK
tokenized_texts_train = [nltk.word_tokenize(text) for text in texts_train]
tokenized_texts_val = [nltk.word_tokenize(text) for text in texts_val]

# Convert tokens to a bag-of-words representation
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([" ".join(tokens) for tokens in tokenized_texts_train])
X_test = vectorizer.transform([" ".join(tokens) for tokens in tokenized_texts_val])

#%% Train a XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Fit the classifier to the training data
xgb_classifier.fit(X_train, labels_train)

# Predict on the test data
predicted_labels_xgb = xgb_classifier.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(labels_val, predicted_labels_xgb)
print(f"Accuracy: {accuracy_xgb:.6f}")
print(f'Number of wrong estimates XGBoost = {sum(labels_val!=predicted_labels_xgb)}')

print(classification_report(labels_val, predicted_labels_xgb, digits=6))
    
plotConfusionMatrix(labels_val = labels_val, labels_pred = predicted_labels_xgb, Title = 'Confusion Matrix XGB-Classifier')

#%% Train a logistic regression classifier
classifier = LogisticRegression(max_iter=10000)  # Increase max_iter value
classifier.fit(X_train, labels_train)

# Predict on the test set
predicted_labels_log = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(labels_val, predicted_labels_log)
print(f"Accuracy: {accuracy:.6f}")
print(f'Number of wrong estimates log reg = {sum(labels_val!=predicted_labels_log)}')

# Display classification report
print(classification_report(labels_val, predicted_labels_log, digits=6))

plotConfusionMatrix(labels_val = labels_val, labels_pred = predicted_labels_log, Title = 'Confusion Matrix logistic regressor')

predictions = predicted_labels_log
#%% make submission

# submission = pd.DataFrame({"id": test_data["id"], "generated": predictions})
# submission_path = r"data\submission.csv" if not is_submission else r"/kaggle/working/submission.csv"
# submission.to_csv(submission_path, index=False)























