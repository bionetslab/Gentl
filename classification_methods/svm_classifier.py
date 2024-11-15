from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
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

# Initialize and train the SVM model
svm_model = SVC(
    kernel='linear', C=100, random_state=42, class_weight='balanced'
    )  # since we have unbalanced data(24,41)
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_model.predict(X_test)
#
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
f1_score_ = f1_score(y_test, y_pred) * 100

print(f"Accuracy for MIBC vs NMIBC: {accuracy:.2f}%")
print(f"F1-score for MIBC vs NMIBC: {f1_score_:.2f}%")
print(classification_report(y_test, y_pred))

"""Accuracy for MIBC vs NMIBC: 84.62%
F1-score for MIBC vs NMIBC: 87.50%"""

#-------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
# csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
# df_cancer_types = pd.read_csv(csv_path)
#
# Dataframe_cancer_with_types = pd.merge(
#     Dataframe_cancer_feature, df_cancer_types[["Final Path", "Anonymized ID"]], left_on='patient_id',
#     right_on="Anonymized ID", how='left'
#     )
#
# Dataframe_cancer_with_types = Dataframe_cancer_with_types.rename(
#     columns={"Final Path": "cancer_type", "Anonymized ID": "patient_id"}
#     )
# Dataframe_cancer_with_types = Dataframe_cancer_with_types.set_index("patient_id")
# Dataframe_cancer_with_types.to_csv("glcm_cancer_features_with_types.csv")
# Dataframe_cancer_with_types["cancer_sub_type_label"] = Dataframe_cancer_with_types["cancer_type"].map(
#     {"T0": 0, "Ta": 1, "Tis": 2, "T1": 3, "T2": 4, "T3": 5, "T4": 6}
#     ).astype(int)
# X = Dataframe_cancer_with_types.drop(
#     columns=["label", "type", "cancer_type", "cancer_sub_type_label"]
#     )  # no need to drop index
# y = Dataframe_cancer_with_types["cancer_sub_type_label"]
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# #
# # Initialize and train the SVM model
# svm_model = SVC(
#     decision_function_shape='ovo', kernel="rbf",C=1,gamma = 0.1, random_state=42, class_weight='balanced'
#     )  # ovo - one vs one
# svm_model.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = svm_model.predict(X_test)
# #
# # Calculate accuracy score
# accuracy = accuracy_score(y_test, y_pred) * 100
# f1_score_ = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems
#
# print(f"Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {accuracy:.2f}%")
# print(f"F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {f1_score_:.2f}%")
# print(classification_report(y_test, y_pred))
"""Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: 30.00%
F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: 16.15%"""
#----------------Plot Decision Boundary-----------------
# DecisionBoundaryDisplay.from_estimator(
#         svm_model,
#         X,
#         response_method="predict",
#         cmap=plt.cm.Spectral,
#         alpha=0.8,
#         xlabel="cancer.feature_names[0]",
#         ylabel="cancer.feature_names[1]",
#     )
#
# # Scatter plot
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolors="k")
# plt.show()

# #-------------------Cancer Vs Non-cancer-----------------------------------------
# full_features_dataframe = pd.concat([Dataframe_cancer_feature, Dataframe_non_cancer_feature], axis=0)
# full_features_dataframe.to_csv("glcm_all_features_svm.csv")
#
# X = full_features_dataframe.drop(columns=["label", "type"])  # no need to drop index
# y = full_features_dataframe["label"]
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# #
# # Initialize and train the SVM model
# svm_model = SVC(kernel='linear',C=1,random_state=42)
# svm_model.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = svm_model.predict(X_test)
# #
# # Calculate accuracy score
# accuracy = accuracy_score(y_test, y_pred) * 100
# f1_score_ = f1_score(y_test, y_pred) * 100
#
# print(f"Accuracy for cancer vs normal ROI: {accuracy:.2f}%")
# print(f"F1-score for cancer vs normal ROI: {f1_score_:.2f}%")
# print(classification_report(y_test, y_pred))
"""Accuracy for cancer vs normal ROI: 100.00%
F1-score for cancer vs normal ROI: 100.00%"""

# #--------------------------T0 Vs T1+-------------------------------------
# Dataframe_cancer_feature["type_label"] = Dataframe_cancer_feature["type"].map({"T0": 0, "T1+": 1})
#
# X = Dataframe_cancer_feature.drop(columns=["label", "type", "type_label"])  # no need to drop index
# y = Dataframe_cancer_feature["type_label"]
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Initialize and train the SVM model
# svm_model = SVC(kernel='linear',C=1,random_state=42)
# svm_model.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = svm_model.predict(X_test)
# #
# # Calculate accuracy score
# accuracy = accuracy_score(y_test, y_pred) * 100
# f1_score_ = f1_score(y_test, y_pred) * 100
#
# print(f"Accuracy for T0 vs T1+: {accuracy:.2f}%")
# print(f"F1-score for T0 vs T1+: {f1_score_:.2f}%")
# print(classification_report(y_test, y_pred))
"""Accuracy for T0 vs T1+: 45.00%
F1-score for T0 vs T1+: 26.67%"""

#---------------Hyperparameter tuning--------------------
# defining parameter range
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}
#
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
#
# # fitting the model for grid search
# grid.fit(X_train, y_train)
#
# # print best parameter after tuning
# # print(grid.best_params_)
#
# # print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)
#
# grid_predictions = grid.predict(X_test)
#
# # print classification report
# print(classification_report(y_test, grid_predictions))
"""Hyperparameter Tuning"""
