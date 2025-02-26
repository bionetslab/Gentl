import numpy as np
import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_methods.best_model.best_model_parameters import load_best_params, append_hyperparams_to_csv, \
    model_evaluation
from classification_methods.features_for_classification import get_features_by_invasion, get_all_features, \
    get_features_by_stage, get_early_late_stage_features, get_features_ptc_vs_mibc, get_tasks


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

    # # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "lda_best_params_new.csv"
        )
    params = format_parameters(best_parameters.get("shrinkage"), best_parameters.get("solver"))
    # # Create LinearDiscriminantAnalysis model
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('lda', LinearDiscriminantAnalysis(**params)
             )]
        )
    #
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = model_evaluation(
        "lda", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task, model, X, y, skf
        )
    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

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

    # # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "lda_best_params_new.csv"
        )
    params = format_parameters(best_parameters.get("shrinkage"), best_parameters.get("solver"))

    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('lda', LinearDiscriminantAnalysis(**params)
             )]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = model_evaluation(
        "lda", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task ,model, X, y, skf
        )
    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

    # Compute average cross-validation scores
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

    # # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "lda_best_params_new.csv"
        )
    params = format_parameters(best_parameters.get("shrinkage"), best_parameters.get("solver"))
    # # Create LinearDiscriminantAnalysis model
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('lda', LinearDiscriminantAnalysis(**params)
             )]
        )
    #
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = model_evaluation(
        "lda", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task ,model, X, y, skf
        )
    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

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

    # # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "lda_best_params_new.csv"
        )
    params = format_parameters(best_parameters.get("shrinkage"), best_parameters.get("solver"))
    # # Create LinearDiscriminantAnalysis model
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('lda', LinearDiscriminantAnalysis(**params)
             )]
        )
    #
    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = model_evaluation(
        "lda", selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task,model, X, y, skf
        )
    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return scores


def hyperparameter_tuning(task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param=None):
    """
    Perform hyperparameter tuning using GridSearchCV with cross-validation.

    Parameters:
        task (str): Task identifier.
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        max_no_of_rois (int): Maximum number of ROIs.
        selected_feature (str): GLCM feature used (e.g., "dissimilarity", "correlation").
        gentl_result_param (str): gentl feature - best distance, max generations, mean distance
        gentl_flag (bool): true if genlt feature is considered

    Returns:
        best_params (dict): Best hyperparameters.
        best_scores (dict): Best cross-validation scores.
    """

    # Define parameter grid with valid solver-shrinkage combinations
    param_grid = [
        {'lda__solver': ['svd']},  # SVD does not support shrinkage

        {'lda__solver': ['lsqr', 'eigen'],  # LSQR and Eigen support shrinkage
         'lda__shrinkage': [None, 'auto', 0.1, 0.5, 1.0]}
        ]

    # Create a pipeline with StandardScaler and LDA
    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),  # Feature scaling
            ('lda', LinearDiscriminantAnalysis())  # LDA model
            ]
        )

    if task == "cancer_stage":
        # Stratified K-Fold Cross-Validation
        stratified_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # GridSearchCV with StratifiedKFold
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=['accuracy', 'f1_macro'],
            refit='f1_macro',  # Optimize based on F1-score
            cv=stratified_k_fold,
            n_jobs=-1,  # Use all available processors
            verbose=3
            )

    else:
        stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # GridSearchCV with StratifiedKFold
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=['accuracy', 'f1_macro'],
            refit='f1_macro',  # Optimize based on F1-score
            cv=stratified_k_fold,
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
        "lda", task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, best_params,
        "lda_best_params_new.csv"
        )
    # # Save best parameters and performance to a text file
    # with open("lda_best_model.txt", "a") as file:
    #     file.write(f"Task: {task} - {max_no_of_rois}\n")
    #     file.write(f"Feature: {selected_feature}\n")
    #     if gentl_flag:
    #         file.write(f"Gentl: {gentl_result_param}\n")
    #     file.write(f"Best Parameters: {best_params}\n")
    #     file.write(f"Best Accuracy: {best_scores['accuracy']:.2f}%\n")
    #     file.write(f"Best F1 Score: {best_scores['f1_score']:.2f}%\n\n")

    return best_params, best_scores


def format_parameters(shrinkage, solver):
    """
    Formats solver and shrinkage parameters for LinearDiscriminantAnalysis.

    Args:
        shrinkage (str or None): Shrinkage value from CSV is returned as a string
        solver (str): Solver type for LDA.

    Returns:
        dict: A dict of keyword arguments for LinearDiscriminantAnalysis.
    """
    if solver == "svd":
        return {"solver": solver}  # SVD does not support shrinkage
    elif pd.isna(shrinkage) or shrinkage in ["nan", "NaN", None]:
        return {"solver": solver, "shrinkage": None}
    elif shrinkage not in ["auto"]:
        return {"solver": solver, "shrinkage": float(shrinkage)}
    else:
        return {"solver": solver, "shrinkage": shrinkage}
