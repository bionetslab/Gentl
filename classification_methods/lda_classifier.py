import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classification_methods.features_for_classification import get_features_by_type

#-------------------NMIBC Vs MIBC----------------------
Dataframe_cancer_with_types = get_features_by_type()

X = Dataframe_cancer_with_types.drop(
    columns=["label", "cancer_type", "cancer_type_label"]
    )  # no need to drop index

y = Dataframe_cancer_with_types["cancer_type_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create LinearDiscriminantAnalysis model
model = LinearDiscriminantAnalysis()

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