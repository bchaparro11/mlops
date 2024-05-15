# Standard Library Imports
import math
import json
import logging

# General Imports
import pandas as pd

# SKLearn Imports

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay

# To receive arguments
from dotenv import load_dotenv
from pathlib import Path
import os

# Importing environment variables
environmentFilePath = Path("./conf.env")
load_dotenv(dotenv_path=environmentFilePath)

# Declaring environment variables
data_location = os.getenv("data_location")
step_1 = os.getenv("step_1")
step_2 = os.getenv("step_2")
step_3 = os.getenv("step_3")
handle_unknown = os.getenv("handle_unknown")
max_iter = int(os.getenv("max_iter"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

# Load Data
# ARGUMENT
df = pd.read_csv(data_location)
# df.info()

# df.head(5)

# df.x6.unique()

# df.x7.unique()

df_X = df.drop("y", axis=1)
df_label = df["y"]

# df_X.head()

numeric_features = ["x1", "x2", "x4", "x5"]
# ARGUMENT --> imputer, median and scaler
numeric_transformer = Pipeline(
    steps=[(step_1, SimpleImputer(strategy=step_2)), (step_3, StandardScaler())]
)

categorical_features = ["x3", "x6", "x7"]
# ARGUMENT --> infrequent_if_exist
categorical_transformer = OneHotEncoder(handle_unknown=handle_unknown)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

#ARGUMENT --> max_iter
clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", LogisticRegression(max_iter=max_iter))]
)

# clf

# Make LogReg Pipeline

# ARGUMENT
RANDOM_STATE=RANDOM_STATE

X_train, X_test, y_train, y_test = train_test_split(
    df_X,
    df_label,
    random_state=RANDOM_STATE
    )

clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))

tprobs = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, clf.predict(X_test)))
print('Confusion matrix:')
print(confusion_matrix(y_test, clf.predict(X_test)))
print(f'AUC: {roc_auc_score(y_test, tprobs)}')
RocCurveDisplay.from_estimator(estimator=clf,X= X_test, y=y_test)