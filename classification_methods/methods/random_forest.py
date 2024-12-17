import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc, max_no_of_rois, tasks


def classify_cancer_invasion():
    # -------------------NMIBC Vs MIBC----------------------
    task = tasks[0]
    Dataframe_cancer_with_types = get_features_by_invasion()

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index
    # X = Dataframe_cancer_with_types.iloc[:, 6:8]
    y = Dataframe_cancer_with_types["cancer_invasion_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test)

    # # Create RandomForest Classifier
    model = RandomForestClassifier(class_weight='balanced', max_depth=5, min_samples_leaf=5, n_estimators=10)
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


def classify_cancer_vs_non_cancerous():
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    task = tasks[1]
    full_features_dataframe = get_all_features()
    X = full_features_dataframe.drop(columns=["label", "cancer_stage"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test)

    # # Create Random Forest classifier
    model = RandomForestClassifier(
        class_weight='balanced', max_depth=10,
        min_samples_leaf=5, n_estimators=200
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


def classify_cancer_stage():
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    task = tasks[2]
    Dataframe_cancer_with_types = get_features_by_stage()
    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test)

    # # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(class_weight='balanced', max_depth=2,
                           min_samples_leaf=100, n_estimators=200)

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


def classify_early_vs_late_stage():
    # ---------------------- Early [Ta,Tis] vs Late Stage [T1,T2,T3,T4]--------------------

    task = tasks[3]
    Dataframe_cancer_with_stages = get_early_late_stage_features()
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test)

    # # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(class_weight='balanced', max_depth=2,
                           min_samples_leaf=20)
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


def classify_ptc_vs_mibc():
    # ---------------------- Post Treatment changes [T0] vs  MIBC [T2,T3,T4]--------------------
    task = tasks[4]
    Dataframe_cancer_with_stages = get_features_ptc_vs_mibc()
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test)

    # # Initialize and train the Random Forest classifier
    model = RandomForestClassifier(class_weight='balanced', max_depth=2,
                           min_samples_leaf=20, n_estimators=10)

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


def hyperparameter_tuning(task, X_train, y_train, X_test, y_test):
    # ------------------- Hyperparameter tuning -------------------

    # Defining parameter range
    param_grid = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
        }

    # Stratified K-Fold Cross-Validation
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV with StratifiedKFold
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced"), param_grid, refit=True, cv=stratified_k_fold, verbose=3
        )
    # grid = GridSearchCV( LogisticRegression(class_weight="balanced", multi_class="ovr"), param_grid, refit=True,
    # cv=stratified_k_fold, verbose=3 )

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
