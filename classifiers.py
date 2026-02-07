import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def preprocess(df, label_col="label"):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le

def train_classifiers(X_train, y_train, X_val, y_val, random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "SVM (RBF)": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[name] = acc

    return results, models
