import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_features_ptc_vs_mibc, get_early_late_stage_features, tasks, max_no_of_rois


def classify_cancer_invasion():
    # #-------------------NMIBC Vs MIBC----------------------
    task = tasks[0]
    Dataframe_cancer_with_types = get_features_by_invasion()

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_invasion_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # best_params = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, rbf_param=True)

    # Initialize and train the SVM model
    model = SVC(
        kernel="rbf",
        C=100, random_state=42, class_weight='balanced'
        )  # since we have unbalanced data(24,41)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # best_param = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, rbf_param=False)

    # Initialize and train the SVM model
    model = SVC(
        decision_function_shape='ovo', kernel="linear", C=0.1, random_state=42, class_weight='balanced'
        )  # ovo - one vs one

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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


def classify_cancer_vs_non_cancerous():
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    task = tasks[1]
    full_features_dataframe = get_all_features()

    X = full_features_dataframe.drop(columns=["label", "cancer_stage"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # hyperparameter_tuning(task, X_train, y_train, X_test, y_test, rbf_param=True)

    # Define the SVM model with a linear kernel
    model = SVC(kernel='rbf', C=1, gamma=1, random_state=42, class_weight='balanced')

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


def classify_early_vs_late_stage():
    # ---------------------- Early [Ta,Tis] vs Late Stage [T1,T2,T3,T4]--------------------
    task = tasks[3]
    Dataframe_cancer_with_stages = get_early_late_stage_features()
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

    # b = hyperparameter_tuning(task, X_train, y_train, X_test, y_test, rbf_param=True)

    # # Initialize and train the SVM model
    model = SVC(kernel='rbf', C=0.1, gamma=0.1, random_state=42, class_weight='balanced')
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

    #hyperparameter_tuning(task, X_train, y_train, X_test, y_test, rbf_param=True)

    # # Initialize and train the SVM model
    model = SVC(kernel='rbf', C=100, gamma=1, random_state=42, class_weight='balanced')

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


def hyperparameter_tuning(task, X_train, y_train, X_test, y_test, rbf_param=False):
    # ------------------- Hyperparameter tuning -------------------
    # Defining parameter range
    if rbf_param:
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
                      'kernel': ['rbf']}
    else:
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}

    # Stratified K-Fold Cross-Validation
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV with StratifiedKFold
    grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, refit=True, cv=stratified_k_fold, verbose=3)
    # grid = GridSearchCV(
    # SVC(class_weight="balanced", decision_function_shape='ovo'), param_grid, refit=True,
    # cv=stratified_k_fold,
    # verbose=3
    # )

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