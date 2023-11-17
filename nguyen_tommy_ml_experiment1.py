"""------------------------------------------------------------------------------------------
# Tommy Nguyen
# Machine Learning
# CSCI 4371
# Experiment 1
------------------------------------------------------------------------------------------"""

### Any packages that are used should be imported here:
#import os
#import csv
import pandas as pd
import numpy as np
#import numpy as np
#from sklearn import linear_model
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def experiment():
  
    # Step 1: Load in data and preprocess
    df = pd.read_csv(r'C:\Users\tommy\Downloads\experiment\Train_data.csv')
    
    # Identify and encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
        # Print the mapping for each column
        print(f"Mapping for column '{col}':")
        for original, encoded in zip(label_encoders[col].classes_, range(len(label_encoders[col].classes_))):
            print(f"{original} => {encoded}")
        print("\n")
    
    # Define features and target variable
    X = df.drop('class', axis=1)  # Features (all columns except 'class')
    y = df['class']  # Labels ('class' column)
    
    # Encode the target variable
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    # Step 2: Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Split training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Step 3: Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)
    
    # Step 4: Evaluate on the validation set
    val_predictions = rf_classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("Validation Accuracy:", val_accuracy)
    
    # Step 5: Retrain the model on the entire training set
    X_full_train = pd.concat([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])
    rf_classifier.fit(X_full_train, y_full_train)
    
    # Step 6: Evaluate the model using the test set
    # Prepare the test set by dropping the target variable 'class' which is to be predicted
    X_test_dropped = X_test.drop('class', axis=1, errors='ignore')
    
    # Evaluate the model using the test set without the target column
    test_predictions = rf_classifier.predict(X_test_dropped)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("Test Accuracy:", test_accuracy)

    # Step 7:
    # Identify misclassified observations in the test set
    misclassified_indices = y_test != test_predictions
    
    # Convert misclassified_indices to a pandas Series to use .loc
    misclassified_indices_series = pd.Series(misclassified_indices, index=X_test.index)
    misclassified_samples_indices = misclassified_indices_series[misclassified_indices_series].index[:5]
    
    # Retrieve the original rows from X_test that were misclassified
    misclassified_samples = X_test.loc[misclassified_samples_indices]
    
    # Add true and predicted labels for clarity
    y_test_series = pd.Series(y_test, index=X_test.index)
    
    misclassified_samples['True Label'] = y_encoder.inverse_transform(y_test_series.loc[misclassified_samples_indices])
    
    test_predictions_series = pd.Series(test_predictions, index=X_test.index)
    
    misclassified_samples['Predicted Label'] = y_encoder.inverse_transform(test_predictions_series.loc[misclassified_samples_indices])
    
    print("Misclassified Test Samples with True Labels:")
    print(misclassified_samples)


if __name__ == '__main__':
    experiment()
