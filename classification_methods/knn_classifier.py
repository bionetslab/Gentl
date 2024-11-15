import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from classification_methods.classification_features import get_features
import pandas as pd

Dataframe_cancer_feature, Dataframe_non_cancer_feature = get_features()

#-------------------NMIBC Vs MIBC----------------------
csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
df_cancer_types = pd.read_csv(csv_path)

Dataframe_cancer_with_types = pd.merge(
    Dataframe_cancer_feature, df_cancer_types[["Final Path", "Anonymized ID"]], left_on='patient_id',
    right_on="Anonymized ID", how='left'
    )

Dataframe_cancer_with_types = Dataframe_cancer_with_types.rename(
    columns={"Final Path": "cancer_type", "Anonymized ID": "patient_id"}
    )
Dataframe_cancer_with_types = Dataframe_cancer_with_types.set_index("patient_id")
Dataframe_cancer_with_types.to_csv("glcm_cancer_features_with_types.csv")
Dataframe_cancer_with_types = Dataframe_cancer_with_types.loc[Dataframe_cancer_with_types['cancer_type'] != 'T0']
Dataframe_cancer_with_types["cancer_type_label"] = Dataframe_cancer_with_types["cancer_type"].map(
    {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    ).astype(int)

X = Dataframe_cancer_with_types.drop(
    columns=["label", "type", "cancer_type", "cancer_type_label"]
    )  # no need to drop index
# X = Dataframe_cancer_with_types.iloc[:, 6:8]
y = Dataframe_cancer_with_types["cancer_type_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create KNN classifier
model = KNeighborsClassifier(n_neighbors=8)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
f1_score_ = f1_score(y_test, y_pred) * 100

print(f"Accuracy for MIBC vs NMIBC: {accuracy:.2f}%")
print(f"F1-score for MIBC vs NMIBC: {f1_score_:.2f}%")
print(classification_report(y_test, y_pred))
"""Accuracy for MIBC vs NMIBC: 76.92%
F1-score for MIBC vs NMIBC: 84.21%"""