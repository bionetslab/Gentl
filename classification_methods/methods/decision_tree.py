import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc


def classify_cancer_invasion():
    # #-------------------NMIBC Vs MIBC----------------------

    Dataframe_cancer_with_stages = get_features_by_invasion()

    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_invasion_label"]

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

    return accuracy, f1_score_


def classify_cancer_vs_non_cancerous():
    # #-------------------Cancer Vs Non-cancer-----------------------------------------

    full_features_dataframe = get_all_features()
    X = full_features_dataframe.drop(columns=["label", "cancer_stage"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight="balanced", random_state=42, criterion='entropy', max_depth=4, min_samples_leaf=5,min_samples_split=18
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

    Dataframe_cancer_with_stages = get_features_by_stage()
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize and train the Decision Tree classifier
    model = DecisionTreeClassifier(class_weight="balanced", random_state=42, max_depth=2, min_samples_leaf=15)
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

    # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight='balanced',max_depth=2, min_samples_leaf=5, random_state=42
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

    # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight='balanced', criterion='entropy', min_samples_split=16,
        max_depth=5, min_samples_leaf=7, random_state=42
        )

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred) * 100

    return accuracy, f1_score_


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
#     }
# # Decision tree classifier
# tree = DecisionTreeClassifier(random_state=1)
# # GridSearchCV
# grid_search = GridSearchCV(
#     estimator=tree, param_grid=param_grid,
#     cv=5, verbose=True
#     )
# grid_search.fit(X_train, y_train)
#
# # Best score and estimator
# print("best accuracy", grid_search.best_score_)
# print(grid_search.best_estimator_)
