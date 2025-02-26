import numpy as np
from sklearn.metrics import make_scorer, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classification_methods.best_model.best_model_parameters import load_best_params, append_hyperparams_to_csv, \
    append_classification_score_to_csv, specificity_score, specificity_score_multiclass, model_evaluation
from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_features_ptc_vs_mibc, get_early_late_stage_features, get_tasks


def classify_cancer_invasion(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
        Performs classification NMIBC Vs MIBC
    Args:
        selected_feature: GLCM feature used (e.g., "dissimilarity", "correlation").
        max_no_of_rois: Maximum number of ROIs 10,20,30,40,50
        gentl_result_param: gentl feature - best distance, max generations, mean distance
        gentl_flag: true if genlt feature is considered

    Returns:
        Accuracy and f1 score
    """

    task = get_tasks()[0]
    Dataframe_cancer_with_types = get_features_by_invasion(
        selected_feature, max_no_of_rois, gentl_result_param, gentl_flag
        )

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_invasion_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_invasion_label"]

    # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param, rbf_param=True
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "svm_best_params_new.csv"
        )
    gamma_value = convert_string_to_float(best_parameters.get("gamma"))
    # Create a pipeline with StandardScaler and SVC
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Make sure scaler is included
            ('svc', SVC(
                kernel=best_parameters.get("kernel"), C=best_parameters.get("C"), gamma=gamma_value,
                random_state=42, class_weight='balanced'
                ))
            ]
        )
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = model_evaluation(
        "svm", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task ,model, X, y, skf
        )

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return scores


def classify_cancer_stage(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
        Performs classification T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4
    Args:
        selected_feature: GLCM feature used (e.g., "dissimilarity", "correlation").
        max_no_of_rois: number of rois considered 10,20,30,40,50
        gentl_result_param: gentl feature - best distance, max generations, mean distance
        gentl_flag: true if genlt feature is considered

    Returns:
        Accuracy and f1 score
    """

    task = get_tasks()[2]
    Dataframe_cancer_with_types = get_features_by_stage(
        selected_feature, max_no_of_rois, gentl_result_param, gentl_flag
        )
    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_stage_label"]

    # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param, rbf_param=True
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "svm_best_params_new.csv"
        )
    gamma_value = convert_string_to_float(best_parameters.get("gamma"))
    # Create an SVM pipeline with the best hyperparameters
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Make sure scaler is included
            ('svc', SVC(
                kernel=best_parameters.get("kernel"), C=best_parameters.get("C"), gamma=gamma_value,
                decision_function_shape='ovo',
                random_state=42, class_weight='balanced'
                ))
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Define scoring metrics for multiclass classification
    scores = model_evaluation(
        "svm", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task ,model, X, y, skf
        )

    # Perform cross-validation
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')

    # Compute average scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return scores


def classify_early_vs_late_stage(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
        Performs classification Early [Ta,Tis] vs Late Stage [T1,T2,T3,T4]
    Args:
        selected_feature: GLCM feature used (e.g., "dissimilarity", "correlation").
        max_no_of_rois: number of rois considered 10,20,30,40,50
        gentl_result_param: gentl feature - best distance, max generations, mean distance
        gentl_flag: true if genlt feature is considered

    Returns:
        Accuracy and f1 score
    """

    task = get_tasks()[3]
    Dataframe_cancer_with_stages = get_early_late_stage_features(
        selected_feature, max_no_of_rois, gentl_result_param, gentl_flag
        )
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param, rbf_param=True
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "svm_best_params_new.csv"
        )
    gamma_value = convert_string_to_float(best_parameters.get("gamma"))
    # Create a pipeline with StandardScaler and SVC
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Make sure scaler is included
            ('svc', SVC(
                kernel=best_parameters.get("kernel"), C=best_parameters.get("C"), gamma=gamma_value,
                random_state=42, class_weight='balanced'
                ))
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = model_evaluation(
        "svm", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task ,model, X, y, skf
        )
    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return scores


def classify_ptc_vs_mibc(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
        Performs classification Post Treatment changes [T0] vs  MIBC [T2,T3,T4]
    Args:
        selected_feature: GLCM feature used (e.g., "dissimilarity", "correlation").
        max_no_of_rois: number of rois considered 10,20,30,40,50
        gentl_result_param: gentl feature - best distance, max generations, mean distance
        gentl_flag: true if genlt feature is considered

    Returns:
        Accuracy and f1 score
    """

    task = get_tasks()[4]
    Dataframe_cancer_with_stages = get_features_ptc_vs_mibc(
        selected_feature, max_no_of_rois, gentl_result_param, gentl_flag
        )
    X = Dataframe_cancer_with_stages.drop(
        columns=["label", "cancer_stage", "cancer_stage_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_stages["cancer_stage_label"]

    # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param, rbf_param=True
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "svm_best_params_new.csv"
        )
    gamma_value = convert_string_to_float(best_parameters.get("gamma"))
    # Create a pipeline with StandardScaler and SVC
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Make sure scaler is included
            ('svc', SVC(
                kernel=best_parameters.get("kernel"), C=best_parameters.get("C"), gamma=gamma_value,
                random_state=42, class_weight='balanced'
                ))
            ]
        )
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = model_evaluation(
        "svm", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task,model, X, y, skf
        )
    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return scores


def hyperparameter_tuning(task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param=None,
                          rbf_param=False):
    """
    Perform hyperparameter tuning using GridSearchCV with cross-validation.

    Parameters:
        gentl_result_param:
        gentl_flag:
        selected_feature:
        task (str): Task identifier.
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        max_no_of_rois (int): Maximum number of ROIs.
        rbf_param (bool): Whether to tune RBF kernel parameters.

    Returns:
        best_params (dict): Best hyperparameters.
        best_scores (dict): Best cross-validation scores.
    """

    # Define parameter grid for SVM
    if rbf_param:
        param_grid = {
            'svc__C': [0.1, 1, 10, 50, 100, 1000],
            'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
            'svc__kernel': ['rbf']
            }
    else:
        param_grid = {
            'svc__C': [0.1, 1, 10, 50, 100, 1000],
            'svc__kernel': ['linear']
            }

    # Create a pipeline with StandardScaler and SVM
    if task == "cancer_stage":
        # Create a pipeline with StandardScaler and SVM
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('svc', SVC(decision_function_shape='ovo', random_state=42, class_weight="balanced"))
                ]
            )

        # Stratified K-Fold Cross-Validation (Ensures all classes are represented in each fold)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Perform GridSearchCV with weighted F1-score for multiclass classification
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            refit='f1_macro',  # Optimize based on F1-score
            scoring=['accuracy', 'f1_macro'],
            cv=skf,
            n_jobs=-1,  # Use all available processors
            verbose=3
            )
    else:
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('svc', SVC(random_state=42, class_weight="balanced"))
                ]
            )

        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform GridSearchCV
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            refit='f1_macro',
            scoring=['accuracy', 'f1_macro'],
            cv=skf,
            n_jobs=-1,  # Use all available processors
            verbose=3
            )

    # Fit GridSearchCV on the entire dataset
    grid.fit(X, y)

    # Get the best hyperparameters and cross-validation scores
    best_params = grid.best_params_
    if task == "cancer_stage":
        best_scores = {
            'accuracy': grid.cv_results_['mean_test_accuracy'][grid.best_index_] * 100,
            'f1_score': grid.cv_results_['mean_test_f1_macro'][grid.best_index_] * 100
            }
    else:
        best_scores = {
            'accuracy': grid.cv_results_['mean_test_accuracy'][grid.best_index_] * 100,
            'f1_score': grid.cv_results_['mean_test_f1_macro'][grid.best_index_] * 100
            }
    append_hyperparams_to_csv(
        "svm", task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, best_params,
        "svm_best_params_new.csv"
        )
    # Save best parameters and performance to a text file
    # with open("svm_best_model.txt", "a") as file:
    #     file.write(f"Task: {task} - {max_no_of_rois}\n")
    #     file.write(f"Feature: {selected_feature}\n")
    #     if gentl_flag:
    #         file.write(f"Gentl: {gentl_result_param}\n")
    #     file.write(f"Best Parameters: {best_params}\n")
    #     file.write(f"Best Accuracy: {best_scores['accuracy']:.2f}%\n")
    #     file.write(f"Best F1 Score: {best_scores['f1_score']:.2f}%\n\n")

    return best_params, best_scores


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
def convert_string_to_float(gamma):
    """
        Converts a string to a float.
    Args:
        gamma: (str)

    Returns:
        gamma as float if its a number
    """
    gamma_value = float(gamma) if gamma not in ["scale", "auto"] else gamma
    return gamma_value
