import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, make_scorer, recall_score
from sklearn.model_selection import cross_validate


def load_best_params(classification_type, glcm_feature, max_no_of_rois, additional_feature, gentl_flag, file_name):
    """
    Load the best hyperparameters for a given classification type, feature, and classifier.

    Parameters:
        max_no_of_rois (int): number of healthy rois per image
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


def append_hyperparams_to_csv(classifier, task, selected_feature, max_no_of_rois, gentl_result_param, gentl_flag,
                              best_params, file_name):
    """
        Append the best hyperparameters for a given classifier.

    Args:
        classifier: name of the classifier
        task: classification task e.g. cancer invasion, cancer stage
        selected_feature: glcm feature
        max_no_of_rois: number of healthy rois per image
        gentl_result_param: genetic algorithm feature - best distance, mean distance, maximum geaneration
        gentl_flag: true if genetic feature is used
        best_params: dictionary of the best parameters from gridsearhcv
        file_name: file to append the parameters
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
            "C": best_params.get("logreg__estimator__C", "") if task == "cancer_stage" else best_params.get(
                "logreg__C", ""
                ),
            "max_iter": best_params.get(
                "logreg__estimator__max_iter", ""
                ) if task == "cancer_stage" else best_params.get("logreg__max_iter", ""),
            "penalty": best_params.get("logreg__estimator__penalty", "") if task == "cancer_stage" else best_params.get(
                "logreg__penalty", ""
                ),
            "solver": best_params.get("logreg__estimator__solver", "") if task == "cancer_stage" else best_params.get(
                "logreg__solver", ""
                ),
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


def append_classification_score_to_csv(classifier, selected_feature, max_no_of_rois, gentl_flag, gentl_result_param,
                                       results, task):
    """
    Append the classification score across all the tasks
    Args:
        task: classification type e.g. cancer invasion
        classifier: classifier type e.g. SVM
        selected_feature: glcm feature
        max_no_of_rois: number of healthy rois per image
        gentl_flag: tru if genetic feature is used
        gentl_result_param: genetic algorithm output - best distance, mean distance, maximum generation
        results: scores to store e.g. f1 score, precision, recall, accuracy, specificity
    """
    new_row = {
        'classifier': classifier,
        "classification_type": task,
        'feature': selected_feature,
        'max_ROI': max_no_of_rois,
        'gentl_result_param': gentl_result_param if gentl_flag else "",
        'accuracy': round(results['accuracy'], 2),
        'f1_score': round(results['f1'], 2),
        'precision': round(results['precision'], 2),
        'sensitivity': round(results['sensitivity'], 2),
        'specificity': round(results['specificity'], 2)
        }

    csv_filename = "classification_performance_results.csv"
    file_path = os.path.join("best_model", csv_filename)
    pd.DataFrame([new_row]).to_csv(file_path, mode='a', header=False, index=False)


def append_classification_score_task_wise_to_csv(classifier, selected_feature, max_no_of_rois, gentl_flag,
                                                    gentl_result_param,
                                                    results, task):
    """
    Append the classification score across all the tasks
    Args:
        task: classification type e.g. cancer invasion
        classifier: classifier type e.g. SVM
        selected_feature: glcm feature
        max_no_of_rois: number of healthy rois per image
        gentl_flag: tru if genetic feature is used
        gentl_result_param: genetic algorithm output - best distance, mean distance, maximum generation
        results: scores to store e.g. f1 score, precision, recall, accuracy, specificity
    """
    gentl_algo_result_param_dict = {"average_best_absolute_distance_results" : "Average best distance",
                                    "average_mean_absolute_distance_results" : "Average mean distance",
                                    "average_generation_absolute_distance_results" : "Average generation"}
    classifier_dict = {
        "svm": "SVM",
        "knn": "KNN",
        "logreg" : "LR",
        "dt": "DT",
        "rf": "RF",
        "lda": "LDA"
        }
    new_row = {
        'Feature': gentl_algo_result_param_dict[gentl_result_param] if gentl_flag else selected_feature,
        "Classifier": classifier_dict[classifier],
        'Accuracy': round(results['accuracy'], 2),
        'F1-score': round(results['f1'], 2),
        'Precision': round(results['precision'], 2),
        'Sensitivity': round(results['sensitivity'], 2),
        'Specificity': round(results['specificity'], 2)
        }

    # Ensure the directory exists
    results_dir = "report_results"
    os.makedirs(results_dir, exist_ok=True)

    # Corrected filename formatting
    csv_filename = f"{task}_{selected_feature}_{max_no_of_rois}_classification_performance_results.csv"
    file_path = os.path.join(results_dir, csv_filename)

    # Check if the file exists to write headers if needed
    file_exists = os.path.isfile(file_path)

    df = pd.DataFrame([new_row])
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def specificity_score_multiclass(y_true, y_pred, num_classes, average='weighted'):
    """
    Computes the specificity for each class in a multi-class classification problem.

    Args:
        y_true (array-like): Ground truth (actual) labels for each sample.
        y_pred (array-like): Predicted labels for each sample.
        num_classes (int): The number of classes.
        average (str): Aggregation method ('weighted' or 'macro').

    Returns:
        float: The averaged specificity score across all classes.
    """
    specificity_scores = []

    for i in range(num_classes):
        tn = np.sum((y_true != i) & (y_pred != i))  # True negatives for class i
        fp = np.sum((y_true != i) & (y_pred == i))  # False positives for class i

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_scores.append(specificity)

    # Compute the final score based on the chosen averaging method
    if average == 'weighted':
        class_counts = np.bincount(y_true, minlength=num_classes)  # Handle missing classes
        weighted_specificity = np.dot(specificity_scores, class_counts) / np.sum(class_counts)
        return weighted_specificity
    elif average == 'macro':
        return np.mean(specificity_scores)  # Simple unweighted average
    else:
        raise ValueError("Invalid value for 'average'. Choose 'macro' or 'weighted'.")


def specificity_score(y_true, y_pred):
    """
    Computes the specificity (True Negative Rate) of a classification model.

    Specificity is defined as the proportion of actual negatives (TN) that are correctly identified:
        Specificity = TN / (TN + FP)

    Args:
        y_true (array-like): Ground truth (actual) binary labels (0 or 1).
        y_pred (array-like): Predicted binary labels (0 or 1) from the classifier.

    Returns:
        float: The specificity score, ranging from 0 to 1. If there are no negative samples, returns 0.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def model_evaluation(classifier, selected_feature, max_no_of_rois, gentl_flag, gentl_result_param, task, model, X, y,
                     skf):
    """
    Compute and store accuracy, f1 score, precision, sensitivity, specificity using cross_validate ()

    Args:
        classifier: type of the classifier e.g. SVM
        selected_feature: glcm feature
        max_no_of_rois: number of healthy rois per image
        gentl_flag: tru if genetic feature is used
        gentl_result_param: genetic algorithm output - best distance, mean distance, maximum generation
        task: classification type e.g. cancer invasion
        model: classifier model with all the hyperparameters
        X: features
        y: labels
        skf: stratified k fold cross validation

    Returns:
        results: classification scores e.g. accuracy, f1 score
    """
    if task == "cancer_stage":
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'specificity': make_scorer(specificity_score_multiclass, num_classes=6, average="macro")
            }
    else:
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'specificity': make_scorer(recall_score, pos_label=0)
            }

    scores = cross_validate(model, X, y, cv=skf, scoring=scoring)

    results = {
        'accuracy': np.mean(scores['test_accuracy']) * 100,
        'f1': np.mean(scores['test_f1_macro']) * 100,
        'precision': np.mean(scores['test_precision_macro']) * 100,
        'sensitivity': np.mean(scores['test_recall_macro']) * 100,
        'specificity': np.mean(scores['test_specificity']) * 100
        }

    # append_classification_score_task_wise_to_csv(
    #     classifier, selected_feature, max_no_of_rois, gentl_flag, gentl_result_param,
    #     results, task
    #     )
    return results
