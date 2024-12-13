import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classification_methods.features_for_classification import get_features_by_type, get_all_features, \
    get_features_by_sub_type


def classify_cancer_invasion():
    # -------------------NMIBC Vs MIBC----------------------
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

    # print(f"Accuracy for MIBC vs NMIBC: {accuracy:.2f}%")
    # print(f"F1-score for MIBC vs NMIBC: {f1_score_:.2f}%")
    # print(classification_report(y_test, y_pred))
    # """Accuracy for MIBC vs NMIBC: 69.23%
    # F1-score for MIBC vs NMIBC: 77.78%"""
    return accuracy, f1_score_


def classify_cancer_vs_non_cancerous():
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    full_features_dataframe = get_all_features()
    X = full_features_dataframe.drop(columns=["label","cancer_type"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #
    # Define LDA model
    model = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred) * 100

    # print(f"Accuracy for cancer vs normal ROI: {accuracy:.2f}%")
    # print(f"F1-score for cancer vs normal ROI: {f1_score_:.2f}%")
    # print(classification_report(y_test, y_pred))
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
    # Initialize and train the Logistic Regression model
    model = LinearDiscriminantAnalysis(shrinkage=0.5, solver='lsqr')
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    #
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems

    # print(f"Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {accuracy:.2f}%")
    # print(f"F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {f1_score_:.2f}%")
    # print(classification_report(y_test, y_pred))
    return accuracy, f1_score_
# Hyperparameter tuning
# # Define the LDA model
# lda = LinearDiscriminantAnalysis()
#
# # Define the hyperparameter grid
# param_grid = {
#     'solver': ['svd', 'lsqr', 'eigen'],  # Options for solvers
#     'shrinkage': [None, 'auto', 0.1, 0.5, 1.0],  # Shrinkage (only for 'lsqr' or 'eigen')
# }
#
# # Perform Grid Search
# grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# # Best parameters and accuracy
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)
#
# # Test the model with the best parameters
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test Set Accuracy:", accuracy)
