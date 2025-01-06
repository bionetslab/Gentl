import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc, get_tasks


def classify_cancer_invasion(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    # -------------------NMIBC Vs MIBC----------------------
    task = get_tasks()[0]
    Dataframe_cancer_with_types = get_features_by_invasion(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag)

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index

    y = Dataframe_cancer_with_types["cancer_invasion_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, max_no_of_rois)

    # Train Logistic regression model
    if selected_feature == "contrast":
        C = 0.2
    else:
        C = 11.288378916846883
    model = LogisticRegression(
            C=C, class_weight='balanced',
            solver='sag', max_iter=1500, random_state=42
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


def classify_cancer_vs_non_cancerous(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    task = get_tasks()[1]
    full_features_dataframe = get_all_features(selected_feature, max_no_of_rois)
    X = full_features_dataframe.drop(columns=["label", "cancer_stage"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, max_no_of_rois)

    # # Train Logistic regression model
    model = LogisticRegression(random_state=42, C=0.08, class_weight='balanced', solver='liblinear')

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


def classify_cancer_stage(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    task = get_tasks()[2]
    Dataframe_cancer_with_types = get_features_by_stage(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag)
    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, max_no_of_rois)

    # # Initialize and train the Logistic Regression model
    base_model = LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=42
        )  # ovr - one vs rest
    model = OneVsRestClassifier(base_model)

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


def classify_early_vs_late_stage(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    # ---------------------- Early [Ta,Tis] vs Late Stage [T1,T2,T3,T4]--------------------
    task = get_tasks()[3]
    Dataframe_cancer_with_stages = get_early_late_stage_features(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag)
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, max_no_of_rois)

    # # Train Logistic regression model
    if selected_feature == "dissimilarity":
        C = 78.47599703514607
    else:
        C = 206
    model = LogisticRegression(
        C=C, class_weight='balanced',
        penalty='l1', solver='liblinear', max_iter=1000, random_state=42
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


def classify_ptc_vs_mibc(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    # ---------------------- Post Treatment changes [T0] vs  MIBC [T2,T3,T4]--------------------

    task = get_tasks()[4]
    Dataframe_cancer_with_stages = get_features_ptc_vs_mibc(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag)
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, max_no_of_rois)

    # # # Train Logistic regression model
    model = LogisticRegression(class_weight="balanced", random_state=42)
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
    param_grid = [
        {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
         'max_iter': [100, 500, 1000, 1500, 2000]
         }
        ]

    # Stratified K-Fold Cross-Validation
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV with StratifiedKFold
    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced"), param_grid, refit=True, cv=stratified_k_fold, verbose=3
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
