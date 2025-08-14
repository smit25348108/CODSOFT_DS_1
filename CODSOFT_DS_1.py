#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
)


# In[4]:


# Replace path if needed
df = pd.read_csv("C:\\Users\\Smit\\OneDrive\\Desktop\\DATASET\\Titanic-Dataset.csv")
df.head()


# In[5]:


print(df.shape)
print(df.isna().sum())
df.describe(include="all").T.head(15)


# In[6]:


target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features]
y = df[target]


# In[7]:


numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

numeric_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_preprocess, numeric_features),
    ("cat", categorical_preprocess, categorical_features)
])


# In[8]:


log_reg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ))
])


# In[9]:


def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    proba = getattr(model, "predict_proba", None)
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_valid, preds))
    print("\nClassification Report:\n", classification_report(y_valid, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_valid, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{model_name} — Confusion Matrix")
    plt.show()

    # ROC-AUC (if probabilistic)
    if proba is not None:
        auc = roc_auc_score(y_valid, proba(X_valid)[:, 1])
        RocCurveDisplay.from_estimator(model, X_valid, y_valid)
        plt.title(f"{model_name} — ROC Curve (AUC = {auc:.3f})")
        plt.show()
    return model

log_model = train_and_evaluate(log_reg, "Logistic Regression")
rf_model  = train_and_evaluate(rf, "Random Forest")


# In[10]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_scores(pipeline, name):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name} CV Accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")

cv_scores(log_reg, "Logistic Regression")
cv_scores(rf, "Random Forest")


# In[11]:


df_fe = df.copy()
df_fe["FamilySize"] = df_fe["SibSp"] + df_fe["Parch"] + 1
df_fe["IsAlone"] = (df_fe["FamilySize"] == 1).astype(int)

features_fe = features + ["FamilySize", "IsAlone"]

X2 = df_fe[features_fe]
y2 = df_fe[target]

num2 = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
cat2 = ["Pclass", "Sex", "Embarked"]

preprocess2 = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num2),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat2),
])

rf_fe = Pipeline([
    ("prep", preprocess2),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])
scores_fe = cross_val_score(rf_fe, X2, y2, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="accuracy", n_jobs=-1)
print(f"RF + FE CV Accuracy: mean={scores_fe.mean():.4f}, std={scores_fe.std():.4f}")
X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
rf_fe.fit(X2_train, y2_train)
y2_pred = rf_fe.predict(X2_valid)
print("Hold-out Accuracy (RF+FE):", accuracy_score(y2_valid, y2_pred))
ConfusionMatrixDisplay(confusion_matrix(y2_valid, y2_pred)).plot()
plt.title("Random Forest + Feature Engineering — Confusion Matrix")
plt.show()


# In[12]:


import joblib
joblib.dump(rf_fe, "titanic_model.joblib")
print("Saved to titanic_model.joblib")


# In[13]:


test_df = pd.read_csv("test.csv")

# Build the same features
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df["IsAlone"] = (test_df["FamilySize"] == 1).astype(int)

X_test_kaggle = test_df[features_fe]  # use the same feature set as the trained model

# Load model (or reuse rf_fe already in memory)
model = joblib.load("titanic_model.joblib")
test_pred = model.predict(X_test_kaggle)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")


# In[ ]:




