from classification_methods.methods.knn_classifier import classify_cancer_invasion as knn_classifier_classify_cancer_invasion, \
    classify_cancer_vs_non_cancerous as knn_classifier_classify_cancer_vs_non_cancerous, \
    classify_cancer_stage as knn_classifier_classify_cancer_stage
from classification_methods.methods.svm_classifier import classify_cancer_invasion as svm_classifier_classify_cancer_invasion, \
    classify_cancer_vs_non_cancerous as svm_classifier_classify_cancer_vs_non_cancerous, \
    classify_cancer_stage as svm_classifier_classify_cancer_stage
from classification_methods.methods.lda_classifier import classify_cancer_invasion as lda_classifier_classify_cancer_invasion, \
    classify_cancer_vs_non_cancerous as lda_classifier_classify_cancer_vs_non_cancerous, \
    classify_cancer_stage as lda_classifier_classify_cancer_stage
from classification_methods.methods.logistic_regression import \
    classify_cancer_invasion as logistic_regression_classify_cancer_invasion, \
    classify_cancer_vs_non_cancerous as logistic_regression_classify_cancer_vs_non_cancerous, \
    classify_cancer_stage as logistic_regression_classify_cancer_stage
from classification_methods.methods.decision_tree import classify_cancer_invasion as decision_tree_classify_cancer_invasion, \
    classify_cancer_vs_non_cancerous as decision_tree_classify_cancer_vs_non_cancerous, \
    classify_cancer_stage as decision_tree_classify_cancer_stage
from classification_methods.methods.random_forest import classify_cancer_invasion as random_forest_classify_cancer_invasion, \
    classify_cancer_vs_non_cancerous as random_forest_classify_cancer_vs_non_cancerous, \
    classify_cancer_stage as random_forest_classify_cancer_stage

def perform_classification():
    """
    Performs classification using the classifiers in the list

    Returns:
    accuracy_dict: Accuracy across classifiers
    f1_score_dict: F1 score across classifiers
    """

    # List of classifiers to use
    classifier_list = ["logistic_regression", "svm_classifier", "knn_classifier",
                       "decision_tree", "random_forest","lda_classifier", ]

    # Mapping classifier methods to task-specific functions
    classifier_methods = {
        "knn_classifier": {
            "cancer_invasion": knn_classifier_classify_cancer_invasion,
            "cancer_non_cancerous": knn_classifier_classify_cancer_vs_non_cancerous,
            "cancer_stage": knn_classifier_classify_cancer_stage
            },
        "svm_classifier": {
            "cancer_invasion": svm_classifier_classify_cancer_invasion,
            "cancer_non_cancerous": svm_classifier_classify_cancer_vs_non_cancerous,
            "cancer_stage": svm_classifier_classify_cancer_stage
            },
        "lda_classifier": {
            "cancer_invasion": lda_classifier_classify_cancer_invasion,
            "cancer_non_cancerous": lda_classifier_classify_cancer_vs_non_cancerous,
            "cancer_stage": lda_classifier_classify_cancer_stage
            },
        "logistic_regression": {
            "cancer_invasion": logistic_regression_classify_cancer_invasion,
            "cancer_non_cancerous": logistic_regression_classify_cancer_vs_non_cancerous,
            "cancer_stage": logistic_regression_classify_cancer_stage
            },
        "decision_tree": {
            "cancer_invasion": decision_tree_classify_cancer_invasion,
            "cancer_non_cancerous": decision_tree_classify_cancer_vs_non_cancerous,
            "cancer_stage": decision_tree_classify_cancer_stage
            },
        "random_forest": {
            "cancer_invasion": random_forest_classify_cancer_invasion,
            "cancer_non_cancerous": random_forest_classify_cancer_vs_non_cancerous,
            "cancer_stage": random_forest_classify_cancer_stage
            }
        }

    # Dictionaries to store results
    accuracy_dict = {
        "cancer_invasion": {},
        "cancer_non_cancerous": {},
        "cancer_stage": {}
        }

    f1_score_dict = {
        "cancer_invasion": {},
        "cancer_non_cancerous": {},
        "cancer_stage": {}
        }

    # Loop through classifiers and tasks
    for method in classifier_list:
        for task, func in classifier_methods[method].items():
            # Call the function and store results
            acc, f1_score = func()
            accuracy_dict[task][method] = round(acc, 2)
            f1_score_dict[task][method] = float(round(f1_score, 2))

    # Print results (optional)
    for task in accuracy_dict:
        print(f"Task: {task}")
        print(f"Accuracy: {accuracy_dict[task]}")
        print(f"F1 Score: {f1_score_dict[task]}")

    return accuracy_dict, f1_score_dict
