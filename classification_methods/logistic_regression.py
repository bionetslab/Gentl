import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classification_methods.features_for_classification import get_features_by_type, get_all_features, \
    get_features_by_sub_type

#-------------------NMIBC Vs MIBC----------------------
Dataframe_cancer_with_types = get_features_by_type()

X = Dataframe_cancer_with_types.drop(
    columns=["label", "cancer_type", "cancer_type_label"]
    )  # no need to drop index
# X = Dataframe_cancer_with_types.iloc[:, 6:8]
y = Dataframe_cancer_with_types["cancer_type_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic regression model
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
f1_score_ = f1_score(y_test, y_pred) * 100

print(f"Accuracy for MIBC vs NMIBC: {accuracy:.2f}%")
print(f"F1-score for MIBC vs NMIBC: {f1_score_:.2f}%")
print(classification_report(y_test, y_pred))

"""Accuracy for MIBC vs NMIBC: 61.54%
F1-score for MIBC vs NMIBC: 66.67%"""

# #-------------------Cancer Vs Non-cancer-----------------------------------------
full_features_dataframe = get_all_features()
X = full_features_dataframe.drop(columns=["label"])  # no need to drop index
y = full_features_dataframe["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic regression model
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
f1_score_ = f1_score(y_test, y_pred) * 100

print(f"Accuracy for cancer vs normal ROI: {accuracy:.2f}%")
print(f"F1-score for cancer vs normal ROI: {f1_score_:.2f}%")
print(classification_report(y_test, y_pred))

# -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
Dataframe_cancer_with_types = get_features_by_sub_type()
X = Dataframe_cancer_with_types.drop(
    columns=["label", "cancer_type", "cancer_sub_type_label"]
    )  # no need to drop index
y = Dataframe_cancer_with_types["cancer_sub_type_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# Initialize and train the Logistic Regression model
model = LogisticRegression(class_weight="balanced",solver="liblinear", multi_class="ovr",random_state=42) # ovr - one vs rest
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
#
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
f1_score_ = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems

print(f"Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {accuracy:.2f}%")
print(f"F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {f1_score_:.2f}%")
print(classification_report(y_test, y_pred))

"""print(classification_report(y_test, y_pred))
Accuracy for T0 vs T1+: 45.00%
F1-score for T0 vs T1+: 15.38%"""
# param_grid = [
#     {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
#      'C': np.logspace(-4, 4, 20),
#      'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
#      'max_iter': [100, 500]
#      }
#     ]
# from sklearn.model_selection import GridSearchCV
#
# clf = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
# best_clf = clf.fit(X_train, y_train)
# print(best_clf.best_estimator_)
# grid_predictions = best_clf.predict(X_test)
#
# # print classification report
# print(classification_report(y_test, grid_predictions))
