from classification_methods.methods.knn_classifier import \
    classify_cancer_invasion as knn_classifier_classify_cancer_invasion, \
    classify_cancer_stage as knn_classifier_classify_cancer_stage, \
    classify_early_vs_late_stage as knn_classifier_classify_early_vs_late_stage, \
    classify_ptc_vs_mibc as knn_classifier_classify_ptc_vs_mibc

from classification_methods.methods.svm_classifier import \
    classify_cancer_invasion as svm_classifier_classify_cancer_invasion, \
    classify_cancer_stage as svm_classifier_classify_cancer_stage, \
    classify_early_vs_late_stage as svm_classifier_classify_early_vs_late_stage, \
    classify_ptc_vs_mibc as svm_classifier_classify_ptc_vs_mibc

from classification_methods.methods.lda_classifier import \
    classify_cancer_invasion as lda_classifier_classify_cancer_invasion, \
    classify_cancer_stage as lda_classifier_classify_cancer_stage, \
    classify_early_vs_late_stage as lda_classifier_classify_early_vs_late_stage, \
    classify_ptc_vs_mibc as lda_classifier_classify_ptc_vs_mibc

from classification_methods.methods.logistic_regression import \
    classify_cancer_invasion as logistic_regression_classify_cancer_invasion, \
    classify_cancer_stage as logistic_regression_classify_cancer_stage, \
    classify_early_vs_late_stage as logistic_regression_classify_early_vs_late_stage, \
    classify_ptc_vs_mibc as logistic_regression_classify_ptc_vs_mibc

from classification_methods.methods.decision_tree import \
    classify_cancer_invasion as decision_tree_classify_cancer_invasion, \
    classify_cancer_stage as decision_tree_classify_cancer_stage, \
    classify_early_vs_late_stage as decision_tree_classify_early_vs_late_stage, \
    classify_ptc_vs_mibc as decision_tree_classify_ptc_vs_mibc

from classification_methods.methods.random_forest import \
    classify_cancer_invasion as random_forest_classify_cancer_invasion, \
    classify_cancer_stage as random_forest_classify_cancer_stage, \
    classify_early_vs_late_stage as random_forest_classify_early_vs_late_stage, \
    classify_ptc_vs_mibc as random_forest_classify_ptc_vs_mibc


def perform_classification(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
    Performs classification using the classifiers in the list

    Returns:
    accuracy_dict: Accuracy across classifiers
    f1_score_dict: F1 score across classifiers
    """

    # List of classifiers to use
    classifier_list = ["logistic_regression", "svm_classifier", "knn_classifier",
                       "decision_tree", "random_forest", "lda_classifier"]

    # Mapping classifier methods to task-specific functions
    classifier_methods = {
        "knn_classifier": {
            "cancer_invasion": knn_classifier_classify_cancer_invasion,
            "cancer_stage": knn_classifier_classify_cancer_stage,
            "cancer_early_vs_late_stage": knn_classifier_classify_early_vs_late_stage,
            "ptc_vs_mibc": knn_classifier_classify_ptc_vs_mibc
            },
        "svm_classifier": {
            "cancer_invasion": svm_classifier_classify_cancer_invasion,
            "cancer_stage": svm_classifier_classify_cancer_stage,
            "cancer_early_vs_late_stage": svm_classifier_classify_early_vs_late_stage,
            "ptc_vs_mibc": svm_classifier_classify_ptc_vs_mibc
            },
        "lda_classifier": {
            "cancer_invasion": lda_classifier_classify_cancer_invasion,
            "cancer_stage": lda_classifier_classify_cancer_stage,
            "cancer_early_vs_late_stage": lda_classifier_classify_early_vs_late_stage,
            "ptc_vs_mibc": lda_classifier_classify_ptc_vs_mibc
            },
        "logistic_regression": {
            "cancer_invasion": logistic_regression_classify_cancer_invasion,
            "cancer_stage": logistic_regression_classify_cancer_stage,
            "cancer_early_vs_late_stage": logistic_regression_classify_early_vs_late_stage,
            "ptc_vs_mibc": logistic_regression_classify_ptc_vs_mibc
            },
        "decision_tree": {
            "cancer_invasion": decision_tree_classify_cancer_invasion,
            "cancer_stage": decision_tree_classify_cancer_stage,
            "cancer_early_vs_late_stage": decision_tree_classify_early_vs_late_stage,
            "ptc_vs_mibc": decision_tree_classify_ptc_vs_mibc
            },
        "random_forest": {
            "cancer_invasion": random_forest_classify_cancer_invasion,
            "cancer_stage": random_forest_classify_cancer_stage,
            "cancer_early_vs_late_stage": random_forest_classify_early_vs_late_stage,
            "ptc_vs_mibc": random_forest_classify_ptc_vs_mibc
            }
        }

    # Dictionaries to store results
    accuracy_dict = {
        "cancer_invasion": {},
        "cancer_stage": {},
        "cancer_early_vs_late_stage": {},
        "ptc_vs_mibc": {}
        }

    f1_score_dict = {
        "cancer_invasion": {},
        "cancer_stage": {},
        "cancer_early_vs_late_stage": {},
        "ptc_vs_mibc": {}
        }

    # Loop through classifiers and tasks
    for method in classifier_list:
        for task, func in classifier_methods[method].items():
            # Call the function and store test accuracy and test f1 score results - use last two
            acc, f1_score, _, _ = func(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag)
            accuracy_dict[task][method] = float(round(acc, 2))
            f1_score_dict[task][method] = float(round(f1_score, 2))

    # Print results
    for task in accuracy_dict:
        print(f"Task: {task}")
        print(f"Accuracy: {accuracy_dict[task]}")
        print(f"F1 Score: {f1_score_dict[task]}")

    return accuracy_dict, f1_score_dict
