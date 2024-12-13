from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from classification_methods.features_for_classification import get_features_by_type, get_all_features, \
    get_features_by_sub_type


def classify_cancer_invasion():
    # #-------------------NMIBC Vs MIBC----------------------
    Dataframe_cancer_with_types = get_features_by_type()

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_type", "cancer_type_label"]
        )  # no need to drop index
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

    # print(f"Accuracy for MIBC vs NMIBC: {accuracy:.2f}%")
    # print(f"F1-score for MIBC vs NMIBC: {f1_score_:.2f}%")
    # print(classification_report(y_test, y_pred))
    # #
    # # """Accuracy for MIBC vs NMIBC: 69.23%
    # # F1-score for MIBC vs NMIBC: 77.78%"""
    return accuracy, f1_score_


def classify_cancer_stage():
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    Dataframe_cancer_with_types = get_features_by_sub_type()
    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_type", "cancer_sub_type_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_sub_type_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #
    # Initialize and train the SVM model
    svm_model = SVC(
        decision_function_shape='ovo', kernel="rbf", C=1, gamma=0.1, random_state=42, class_weight='balanced'
        )  # ovo - one vs one
    svm_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_model.predict(X_test)
    #
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems

    # print(f"Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {accuracy:.2f}%")
    # print(f"F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {f1_score_:.2f}%")
    # print(classification_report(y_test, y_pred))
    # """Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: 30.00%
    # F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: 16.15%"""
    return accuracy, f1_score_


def classify_cancer_vs_non_cancerous():
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    full_features_dataframe = get_all_features()

    X = full_features_dataframe.drop(columns=["label","cancer_type"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #
    # Initialize and train the SVM model
    svm_model = SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_model.predict(X_test)
    #
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred) * 100

    # print(f"Accuracy for cancer vs normal ROI: {accuracy:.2f}%")
    # print(f"F1-score for cancer vs normal ROI: {f1_score_:.2f}%")
    # print(classification_report(y_test, y_pred))
    # """Accuracy for cancer vs normal ROI: 72.50%
    # F1-score for cancer vs normal ROI: 77.55%"""
    return accuracy, f1_score_


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
