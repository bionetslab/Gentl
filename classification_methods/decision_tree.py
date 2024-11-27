import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# Create Decision Tree classifier
model = DecisionTreeClassifier(
    class_weight='balanced', criterion='entropy',
    max_depth=2, min_samples_leaf=7, random_state=42
    )

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
"""Accuracy for MIBC vs NMIBC: 69.23%
F1-score for MIBC vs NMIBC: 75.00%"""

# #-------------------Cancer Vs Non-cancer-----------------------------------------
full_features_dataframe = get_all_features()
X = full_features_dataframe.drop(columns=["label"])  # no need to drop index
y = full_features_dataframe["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Decision Tree classifier
model = DecisionTreeClassifier(class_weight="balanced", random_state=42,criterion='entropy', max_depth=3, min_samples_leaf=17)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
f1_score_ = f1_score(y_test, y_pred) * 100

print(f"Accuracy for cancer vs normal ROI: {accuracy:.2f}%")
print(f"F1-score for cancer vs normal ROI: {f1_score_:.2f}%")
print(classification_report(y_test, y_pred))
"""Accuracy for cancer vs normal ROI: 62.50%
F1-score for cancer vs normal ROI: 68.09%
"""

# -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
Dataframe_cancer_with_types = get_features_by_sub_type()
X = Dataframe_cancer_with_types.drop(
    columns=["label", "cancer_type", "cancer_sub_type_label"]
    )  # no need to drop index
y = Dataframe_cancer_with_types["cancer_sub_type_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the Decision Tree classifier
model = DecisionTreeClassifier(class_weight="balanced", random_state=42,max_depth=2, min_samples_leaf=15)
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

# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
# # Plot the decision tree
# plt.figure(figsize=(20, 10))
# plot_tree(model, filled=True, feature_names=X.columns, class_names=["NMIBC","MIBC"])
# plt.show()

# from sklearn.model_selection import GridSearchCV
#
# # Hyperparameter to fine tune
# param_grid = {
#     'max_depth': range(1, 10, 1),
#     'min_samples_leaf': range(1, 20, 2),
#     'min_samples_split': range(2, 20, 2),
#     'criterion': ["entropy", "gini"]
# }
# # Decision tree classifier
# tree = DecisionTreeClassifier(random_state=1)
# # GridSearchCV
# grid_search = GridSearchCV(estimator=tree, param_grid=param_grid,
#                            cv=5, verbose=True)
# grid_search.fit(X_train, y_train)
#
# # Best score and estimator
# print("best accuracy", grid_search.best_score_)
# print(grid_search.best_estimator_)