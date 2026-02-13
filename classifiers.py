import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def preprocess(df, label_col="label", test_size=0.2, val_size=0.25, random_state=42):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )

    return X_train, y_train, X_test, y_test, le

def train_classifiers(X_train, y_train, X_val, y_val, random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10_000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM (RBF)": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=10_000, random_state=random_state)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[name] = acc

    return results, models

def classify_datapoint(model, X):
    pred = model.predict(X)
    return pred
