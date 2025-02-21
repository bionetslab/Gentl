import pandas as pd
import os


def load_best_params(classification_type, glcm_feature, max_no_of_rois, additional_feature, gentl_flag, file_name):
    """
    Load the best hyperparameters for a given classification type, feature, and classifier.

    Parameters:
        max_no_of_rois:
        classification_type (str): Type of classification (e.g., "cancer invasion", "stage classification").
        glcm_feature (str): GLCM feature used (e.g., "dissimilarity", "correlation").
        additional_feature (str): Additional feature used if gentl_flag is True.
        gentl_flag (bool): Whether to consider the additional_feature column.
        file_name (str): Classifier-specific CSV file name.

    Returns:
        dict: Best parameters for the given classifier and features.
    """
    file_path = os.path.join("best_model", file_name)
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert empty additional_feature values to None (for consistent handling)
    df["additional_feature"] = df["additional_feature"].fillna("")

    filtered_df = df[
        (df["classification_type"] == classification_type) &
        (df["glcm_feature"] == glcm_feature) &
        (df["max_no_of_rois"] == max_no_of_rois)
        ]

    # If gentl_flag is True, filter using additional_feature
    if gentl_flag:
        filtered_df = filtered_df[filtered_df["additional_feature"] == additional_feature]
    else:
        # If gentl_flag is False, select rows where additional_feature is empty
        filtered_df = filtered_df[filtered_df["additional_feature"] == ""]

    # Extract the best parameters based on classifier type
    best_params = filtered_df.iloc[0].to_dict()

    # Remove metadata columns
    best_params.pop("classification_type", None)
    best_params.pop("glcm_feature", None)
    best_params.pop("additional_feature", None)  # Always remove additional_feature after filtering

    return best_params


def append_hyperparams_to_csv(classifier, task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag,best_params, file_name):
    """
    Appends hyperparameter tuning results to a CSV file based on the model type.
    """

    file_path = os.path.join("best_model", file_name)

    # Define column structure based on model type
    if classifier == "svm":
        new_row = {
            "classification_type": task,
            "glcm_feature": selected_feature,
            "additional_feature": gentl_result_param if gentl_flag else "",
            "C": best_params.get("svc__C", ""),
            "gamma": best_params.get("svc__gamma", ""),
            "kernel": best_params.get("svc__kernel", ""),
            "max_no_of_rois": max_no_of_rois
            }

    elif classifier == "knn":
        new_row = {
            "classification_type": task,
            "glcm_feature": selected_feature,
            "additional_feature": gentl_result_param if gentl_flag else "",
            "metric": best_params.get("knn__metric", ""),
            "n_neighbors": best_params.get("knn__n_neighbors", ""),
            "p": best_params.get("knn__p", ""),
            "weights": best_params.get("knn__weights", ""),
            "max_no_of_rois": max_no_of_rois
            }
    elif classifier == "logistic_regression":
        new_row = {
            "classification_type": task,
            "glcm_feature": selected_feature,
            "additional_feature": gentl_result_param if gentl_flag else "",
            "C": best_params.get("logreg__estimator__C", "") if task == "cancer_stage" else best_params.get("logreg__C", ""),
            "max_iter": best_params.get("logreg__estimator__max_iter", "") if task == "cancer_stage" else best_params.get("logreg__max_iter", ""),
            "penalty": best_params.get("logreg__estimator__penalty", "") if task == "cancer_stage" else best_params.get("logreg__penalty", ""),
            "solver": best_params.get("logreg__estimator__solver", "") if task == "cancer_stage" else best_params.get("logreg__solver", ""),
            "max_no_of_rois": max_no_of_rois
            }
    elif classifier == "lda":
        new_row = {
            "classification_type": task,
            "glcm_feature": selected_feature,
            "additional_feature": gentl_result_param if gentl_flag else "",
            "shrinkage": best_params.get("lda__shrinkage", ""),
            "solver": best_params.get("lda__solver", ""),
            "max_no_of_rois": max_no_of_rois
            }
    elif classifier == "decision_tree":
        new_row = {
            "classification_type": task,
            "glcm_feature": selected_feature,
            "additional_feature": gentl_result_param if gentl_flag else "",
            "criterion": best_params.get("dt__criterion", ""),
            "max_depth": best_params.get("dt__max_depth", ""),
            "min_samples_leaf": best_params.get("dt__min_samples_leaf", ""),
            "min_samples_split": best_params.get("dt__min_samples_split", ""),
            "max_no_of_rois": max_no_of_rois
            }
    elif classifier == "random_forest":
        new_row = {
            "classification_type": task,
            "glcm_feature": selected_feature,
            "additional_feature": gentl_result_param if gentl_flag else "",
            "criterion": best_params.get("rf__criterion", ""),
            "max_depth": best_params.get("rf__max_depth", ""),
            "max_features": best_params.get("rf__max_features", ""),
            "min_samples_leaf": best_params.get("rf__min_samples_leaf", ""),
            "min_samples_split": best_params.get("rf__min_samples_split", ""),
            "n_estimators": best_params.get("rf__n_estimators", ""),
            "max_no_of_rois": max_no_of_rois
            }
    else:
        pass
    # Convert new_row to DataFrame and append
    pd.DataFrame([new_row]).to_csv(file_path, mode='a', header=False, index=False)

    print(f"Hyperparameter tuning results for {classifier.upper()} appended successfully.")


