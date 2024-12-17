import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc


def classify_cancer_invasion():
    # -------------------NMIBC Vs MIBC----------------------
    Dataframe_cancer_with_types = get_features_by_invasion()

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index

    y = Dataframe_cancer_with_types["cancer_invasion_label"]

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

    return accuracy, f1_score_


def classify_cancer_stage():
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    Dataframe_cancer_with_types = get_features_by_stage()
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

    return accuracy, f1_score_

def classify_early_vs_late_stage():
    # ---------------------- Early [Ta,Tis] vs Late Stage [T1,T2,T3,T4]--------------------

    Dataframe_cancer_with_stages = get_early_late_stage_features()
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

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

    return accuracy, f1_score_


def classify_ptc_vs_mibc():
    # ---------------------- Post Treatment changes [T0] vs  MIBC [T2,T3,T4]--------------------

    Dataframe_cancer_with_stages = get_features_ptc_vs_mibc()
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

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
