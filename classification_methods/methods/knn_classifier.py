import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_methods.best_model.best_model_parameters import load_best_params, append_hyperparams_to_csv
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
    # X = Dataframe_cancer_with_types.iloc[:, 6:8]
    y = Dataframe_cancer_with_types["cancer_invasion_label"]

    # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "knn_best_params.csv"
        )
    # Define KNN model with scaling using a pipeline
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('knn', KNeighborsClassifier(
                n_neighbors=best_parameters.get("n_neighbors"), weights=best_parameters.get("weights"),
                metric=best_parameters.get("metric"),p=best_parameters.get("p")
                ))  # KNN model
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return avg_accuracy, avg_f1, 0, 0


def classify_cancer_vs_non_cancerous(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
           Performs classification Cancer Vs Non-cancer
       Args:
           selected_feature: GLCM feature used (e.g., "dissimilarity", "correlation").
           max_no_of_rois: number of rois considered 10,20,30,40,50
           gentl_result_param: gentl feature - best distance, max generations, mean distance
           gentl_flag: true if genlt feature is considered

       Returns:
           Accuracy and f1 score
    """

    task = get_tasks()[1]
    full_features_dataframe = get_all_features(selected_feature, max_no_of_rois)
    X = full_features_dataframe.drop(columns=["label", "cancer_stage"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Perform hyperparameter tuning
    # best_params, best_scores = hyperparameter_tuning(
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "knn_best_params.csv"
        )
    # Define KNN model with scaling using a pipeline
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('knn', KNeighborsClassifier(
                n_neighbors=best_parameters.get("n_neighbors"), weights=best_parameters.get("weights"),
                metric=best_parameters.get("metric"),p=best_parameters.get("p")
                ))  # KNN model
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return avg_accuracy, avg_f1, 0, 0


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
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "knn_best_params.csv"
        )
    # Define KNN model with scaling using a pipeline
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('knn', KNeighborsClassifier(
                n_neighbors=best_parameters.get("n_neighbors"), weights=best_parameters.get("weights"),
                metric=best_parameters.get("metric"),p=best_parameters.get("p")
                ))  # KNN model
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return avg_accuracy, avg_f1, 0, 0


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
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "knn_best_params.csv"
        )
    # Define KNN model with scaling using a pipeline
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('knn', KNeighborsClassifier(
                n_neighbors=best_parameters.get("n_neighbors"), weights=best_parameters.get("weights"),
                metric=best_parameters.get("metric"),p=best_parameters.get("p")
                ))  # KNN model
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return avg_accuracy, avg_f1, 0, 0


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
    #     task, X, y, max_no_of_rois, selected_feature, gentl_flag, gentl_result_param
    #     )
    # print(task)
    # print("Best Parameters:", best_params)
    # print("Best Scores:", best_scores)
    best_parameters = load_best_params(
        task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, "knn_best_params.csv"
        )
    # Define KNN model with scaling using a pipeline
    model = Pipeline(
        [
            ('scaler', StandardScaler()),  # Normalize features
            ('knn', KNeighborsClassifier(
                n_neighbors=best_parameters.get("n_neighbors"), weights=best_parameters.get("weights"),
                metric=best_parameters.get("metric"),p=best_parameters.get("p")
                ))  # KNN model
            ]
        )

    # Define Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and compute scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    # Compute average cross-validation scores
    avg_accuracy = np.mean(accuracy_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    return avg_accuracy, avg_f1, 0, 0


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

    # Define the parameter grid for KNN
    param_grid = {
        'knn__n_neighbors': np.arange(2, 30, 1),  # Number of neighbors
        'knn__weights': ['uniform', 'distance'],  # Weight function
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
        'knn__p': [1, 2]  # Minkowski power parameter (1=Manhattan, 2=Euclidean)
        }

    if task == "cancer_stage":
        # Create a pipeline with StandardScaler and KNN
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),  # Normalize features
                ('knn', KNeighborsClassifier())  # KNN model
                ]
            )

        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Perform GridSearchCV
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=['accuracy', 'f1_weighted'],
            refit='f1_weighted',  # Optimize based on F1-score
            cv=skf,
            n_jobs=-1,  # Use all available processors
            verbose=3
            )
    else:
        # Create a pipeline with StandardScaler and KNN
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),  # Normalize features
                ('knn', KNeighborsClassifier())  # KNN model
                ]
            )

        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform GridSearchCV
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=['accuracy', 'f1'],
            refit='f1',  # Optimize based on F1-score
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
            'f1_score': grid.cv_results_['mean_test_f1_weighted'][grid.best_index_] * 100
            }
    else:
        best_scores = {
            'accuracy': grid.cv_results_['mean_test_accuracy'][grid.best_index_] * 100,
            'f1_score': grid.cv_results_['mean_test_f1'][grid.best_index_] * 100
            }
    # append_hyperparams_to_csv(
    #     "knn", task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag, best_params,
    #     "knn_best_params.csv"
    #     )
    # # Save best parameters and performance to a text file
    # with open("knn_best_model.txt", "a") as file:
    #     file.write(f"Task: {task} - {max_no_of_rois}\n")
    #     file.write(f"Feature: {selected_feature}\n")
    #     if gentl_flag:
    #         file.write(f"Gentl: {gentl_result_param}\n")
    #     file.write(f"Best Parameters: {best_params}\n")
    #     file.write(f"Best Accuracy: {best_scores['accuracy']:.2f}%\n")
    #     file.write(f"Best F1 Score: {best_scores['f1_score']:.2f}%\n\n")

    return best_params, best_scores
