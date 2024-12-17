import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc


def classify_cancer_invasion():
    # -------------------NMIBC Vs MIBC----------------------
    Dataframe_cancer_with_types = get_features_by_invasion()

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_type", "cancer_type_label"]
        )  # no need to drop index
    # X = Dataframe_cancer_with_types.iloc[:, 6:8]
    y = Dataframe_cancer_with_types["cancer_type_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create RandomForest Classifier
    model = RandomForestClassifier(
        class_weight='balanced', max_depth=2,
        min_samples_leaf=20, n_estimators=200, random_state=42
        )

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
    X = full_features_dataframe.drop(columns=["label", "cancer_type"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create Random Forest classifier
    model = RandomForestClassifier(
        class_weight='balanced', max_depth=10,
        min_samples_leaf=5, n_estimators=10, random_state=42
        )

    # Train the model
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(
        class_weight="balanced", random_state=42, max_depth=3, min_samples_leaf=5,
        n_estimators=25  # number of trees
        )
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

    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(
        class_weight='balanced', max_depth=2,
        min_samples_leaf=20, n_estimators=200, random_state=42
        )

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

    # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(
        class_weight='balanced', max_depth=2,
        min_samples_leaf=20, n_estimators=200, random_state=42
        )

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred) * 100

    return accuracy, f1_score_
# from sklearn.model_selection import GridSearchCV
#
# # Hyperparameter to fine tune
# param_grid = {
#     'max_depth': [2,3,5,10,20],
#     'min_samples_leaf': [5,10,20,50,100,200],
#     'n_estimators': [10,25,30,50,100,200]
# }
# # Decision tree classifier
# tree = RandomForestClassifier(random_state=42,class_weight="balanced")
# # GridSearchCV
# grid_search = GridSearchCV(estimator=tree, param_grid=param_grid,
#                            cv=5, verbose=True)
# grid_search.fit(X_train, y_train)
#
# # Best score and estimator
# print("best accuracy", grid_search.best_score_)
# print(grid_search.best_estimator_)
