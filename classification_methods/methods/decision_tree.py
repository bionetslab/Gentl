import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc, get_tasks


def classify_cancer_invasion(selected_feature, max_no_of_rois):
    # #-------------------NMIBC Vs MIBC----------------------
    task = get_tasks()[0]
    Dataframe_cancer_with_stages = get_features_by_invasion(selected_feature, max_no_of_rois)

    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_invasion_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # hyperparameter_tuning(task, X_train, y_train, X_test, y_test,max_no_of_rois)

    # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight='balanced', criterion='entropy',
        max_depth=1, min_samples_leaf=19, min_samples_split=6, random_state=42
        )
    #
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    print(f"Cross-Validation Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Cross-Validation Average F1-Score: {avg_f1:.2f}%")

    # Train on the full training/validation set
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    test_f1 = f1_score(y_test, y_pred) * 100

    print(f"Test Set Accuracy: {test_accuracy:.2f}%")
    print(f"Test Set F1-Score: {test_f1:.2f}%")
    print(classification_report(y_test, y_pred))

    return avg_accuracy, avg_f1, test_accuracy, test_f1


def classify_cancer_vs_non_cancerous(selected_feature, max_no_of_rois):
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    task = get_tasks()[1]
    full_features_dataframe = get_all_features(selected_feature, max_no_of_rois)
    X = full_features_dataframe.drop(columns=["label", "cancer_stage"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # hyperparameter_tuning(task, X_train, y_train, X_test, y_test,max_no_of_rois)

    # # Create Decision Tree classifier
    model = DecisionTreeClassifier(class_weight='balanced', max_depth=7)
    #
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    print(f"Cross-Validation Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Cross-Validation Average F1-Score: {avg_f1:.2f}%")

    # Train on the full training/validation set
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    test_f1 = f1_score(y_test, y_pred) * 100

    print(f"Test Set Accuracy: {test_accuracy:.2f}%")
    print(f"Test Set F1-Score: {test_f1:.2f}%")
    print(classification_report(y_test, y_pred))

    return avg_accuracy, avg_f1, test_accuracy, test_f1


def classify_cancer_stage(selected_feature, max_no_of_rois):
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    task = get_tasks()[2]
    Dataframe_cancer_with_stages = get_features_by_stage(selected_feature, max_no_of_rois)
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # hyperparameter_tuning(task, X_train, y_train, X_test, y_test,max_no_of_rois)

    # Initialize and train the Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight='balanced', criterion='entropy',
        max_depth=7
        )
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    print(f"Cross-Validation Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Cross-Validation Average F1-Score: {avg_f1:.2f}%")

    # Train on the full training/validation set
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    test_f1 = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems

    print(f"Test Set Accuracy: {test_accuracy:.2f}%")
    print(f"Test Set F1-Score: {test_f1:.2f}%")
    print(classification_report(y_test, y_pred))

    return avg_accuracy, avg_f1, test_accuracy, test_f1


def classify_early_vs_late_stage(selected_feature, max_no_of_rois):
    # ---------------------- Early [Ta,Tis] vs Late Stage [T1,T2,T3,T4]--------------------
    task = get_tasks()[3]
    Dataframe_cancer_with_stages = get_early_late_stage_features(selected_feature, max_no_of_rois)
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # hyperparameter_tuning(task, X_train, y_train, X_test, y_test,max_no_of_rois)

    # # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight='balanced', max_depth=2,
        min_samples_leaf=11
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    print(f"Cross-Validation Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Cross-Validation Average F1-Score: {avg_f1:.2f}%")

    # Train on the full training set
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    test_f1 = f1_score(y_test, y_pred) * 100

    print(f"Test Set Accuracy: {test_accuracy:.2f}%")
    print(f"Test Set F1-Score: {test_f1:.2f}%")
    print(classification_report(y_test, y_pred))

    return avg_accuracy, avg_f1, test_accuracy, test_f1


def classify_ptc_vs_mibc(selected_feature, max_no_of_rois):
    # ---------------------- Post Treatment changes [T0] vs  MIBC [T2,T3,T4]--------------------
    task = get_tasks()[4]
    Dataframe_cancer_with_stages = get_features_ptc_vs_mibc(selected_feature, max_no_of_rois)
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # hyperparameter_tuning(task, X_train, y_train, X_test, y_test,max_no_of_rois)

    # # Create Decision Tree classifier
    model = DecisionTreeClassifier(
        class_weight='balanced', criterion='entropy',
        max_depth=4, min_samples_leaf=3, min_samples_split=4
        )
    #
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    print(f"Cross-Validation Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Cross-Validation Average F1-Score: {avg_f1:.2f}%")

    # Train on the full training set
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    test_f1 = f1_score(y_test, y_pred) * 100

    print(f"Test Set Accuracy: {test_accuracy:.2f}%")
    print(f"Test Set F1-Score: {test_f1:.2f}%")
    print(classification_report(y_test, y_pred))

    return avg_accuracy, avg_f1, test_accuracy, test_f1


def hyperparameter_tuning(task, X_train, y_train, X_test, y_test, max_no_of_rois):
    # ------------------- Hyperparameter tuning -------------------

    # Defining parameter range
    param_grid = {
        'max_depth': range(1, 10, 1),
        'min_samples_leaf': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'criterion': ["entropy", "gini"]
        }

    # Stratified K-Fold Cross-Validation
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV with StratifiedKFold
    grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced"), param_grid, refit=True, cv=stratified_k_fold, verbose=3
        )

    # Fitting the model for grid search
    grid.fit(X_train, y_train)

    # Best parameters and best estimator
    best_params = grid.best_params_
    best_estimator = grid.best_estimator_

    # Predictions on the test set
    grid_predictions = grid.predict(X_test)

    # Accuracy on test set
    accuracy = accuracy_score(y_test, grid_predictions)

    # Print results
    print("Task:", task)
    print("Best Estimator:", best_estimator)
    print("Test Accuracy:", round(accuracy * 100, 2))
    print("Classification Report on Test Data:\n", classification_report(y_test, grid_predictions))
    # print(grid.cv_results_)

    # Save results to a text file
    with open("best_model.txt", "a") as file:
        file.write("Task:\n")
        file.write(f"{task} - {max_no_of_rois}\n\n")
        file.write("Best Estimator:\n")
        file.write(f"{best_estimator}\n\n")
        file.write("Test Accuracy:\n")
        file.write(f"{round(accuracy * 100, 2)}\n\n")
    return grid.best_params_
