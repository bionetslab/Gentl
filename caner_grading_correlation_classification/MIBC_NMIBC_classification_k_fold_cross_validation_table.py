import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, make_scorer, classification_report)
from sklearn.utils import class_weight

# Global variable for cancer grade order
cancer_grade_order = ['T0', 'Ta', 'Tis', 'T1', 'T2', 'T3', 'T4']

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0


def evaluate_model_cv_all(model, X, y, model_name, feature_column, cv=10):
    """
    Evaluate a model using 10-fold cross validation for multiple metrics.
    Returns a dictionary with keys: 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1'.
    Each value is a tuple (mean, std) in percentage.
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score),
        'f1': 'f1_weighted'
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    results = {}
    for key in scoring.keys():
        scores = cv_results[f'test_{key}']
        mean_score = np.mean(scores) * 100
        std_score = np.std(scores) * 100
        print(f"{model_name} based on {feature_column} - {key}: {mean_score:.2f}% ± {std_score:.2f}%")
        results[key] = (mean_score, std_score)
    return results


# Classifier functions using multiple-metric CV
def decision_tree_cv_all(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=2, min_samples_leaf=7, random_state=42)
    return evaluate_model_cv_all(model, X, y, "Decision Tree", feature_column)


def knn_cv_all(merged_df, feature_column, n_neighbors=5):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return evaluate_model_cv_all(model, X, y, "KNN", feature_column)


def lda_cv_all(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = LinearDiscriminantAnalysis()
    return evaluate_model_cv_all(model, X, y, "LDA", feature_column)


def logistic_regression_cv_all(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = LogisticRegression(max_iter=1000, random_state=42)
    return evaluate_model_cv_all(model, X, y, "Logistic Regression", feature_column)


def random_forest_cv_all(merged_df, feature_column, n_estimators=100):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(cw))
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight_dict, random_state=42)
    return evaluate_model_cv_all(model, X, y, "Random Forest", feature_column)


def svm_cv_all(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = SVC(kernel='linear', C=1.0, random_state=42)
    return evaluate_model_cv_all(model, X, y, "SVM", feature_column)


def run_classifiers_cv_all(merged_df, feature_column):
    scores = {}
    scores["Decision Tree"] = decision_tree_cv_all(merged_df, feature_column)
    scores["KNN"] = knn_cv_all(merged_df, feature_column)
    scores["LDA"] = lda_cv_all(merged_df, feature_column)
    scores["Logistic Regression"] = logistic_regression_cv_all(merged_df, feature_column)
    scores["Random Forest"] = random_forest_cv_all(merged_df, feature_column)
    scores["SVM"] = svm_cv_all(merged_df, feature_column)
    return scores


def process_file(file_path, metric_type, glcm_feature, cancer_grades_df):
    """
    Load the average file (e.g., average_best_distance, average_mean_distance, or average_generation),
    merge with cancer grades, and return the merged dataframe along with the feature column name.
    """
    avg_df = pd.read_csv(file_path)
    merged_df = avg_df.merge(cancer_grades_df, on='patient_id', how='left')
    merged_df['cancer_grade'] = pd.Categorical(merged_df['cancer_grade'],
                                               categories=cancer_grade_order, ordered=True)
    if metric_type == "average_generation":
        feature_column = "average_generation"
    elif metric_type in ["average_best_distance", "average_mean_distance"]:
        if "average_distance" in merged_df.columns:
            feature_column = "average_distance"
        else:
            raise ValueError(f"Column 'average_distance' not found in {file_path}.")
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    return merged_df, feature_column


def generate_table(results, metric_type):
    """
    Generate a table (DataFrame) with headers:
    Machine Learning Method, GLCM Feature, Accuracy, Sensitivity, Specificity, Precision, F1-score.
    Each cell contains "mean ± std" for the corresponding metric.
    """
    rows = []
    ml_methods = list(results[metric_type].keys())
    glcm_features = list(next(iter(results[metric_type].values())).keys())
    for ml in ml_methods:
        for feature in glcm_features:
            metrics = results[metric_type][ml][
                feature]  # a dict with keys: accuracy, sensitivity, specificity, precision, f1
            row = {
                "Machine Learning Method": ml,
                "GLCM Feature": feature,
                "Accuracy": f"{metrics['accuracy'][0]:.2f} ± {metrics['accuracy'][1]:.2f}",
                "Sensitivity": f"{metrics['sensitivity'][0]:.2f} ± {metrics['sensitivity'][1]:.2f}",
                "Specificity": f"{metrics['specificity'][0]:.2f} ± {metrics['specificity'][1]:.2f}",
                "Precision": f"{metrics['precision'][0]:.2f} ± {metrics['precision'][1]:.2f}",
                "F1-score": f"{metrics['f1'][0]:.2f} ± {metrics['f1'][1]:.2f}"
            }
            rows.append(row)
    df_table = pd.DataFrame(rows)
    return df_table


# ===================== Main =====================
if __name__ == "__main__":
    # Step 1: Load cancer grades file
    cancer_grades_df = pd.read_csv(
        '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv')
    cancer_grades_df.rename(columns={'Anonymized ID': 'patient_id', 'Final Path': 'cancer_grade'}, inplace=True)
    cancer_grades_df.dropna(subset=['patient_id', 'cancer_grade'], inplace=True)
    cancer_grades_df = cancer_grades_df[['patient_id', 'cancer_grade']]

    # Define ML methods, GLCM features, and metric types.
    ML_METHODS = ["Decision Tree", "KNN", "LDA", "Logistic Regression", "Random Forest", "SVM"]
    GLCM_FEATURES = ["homogeneity", "contrast", "correlation", "dissimilarity", "energy"]
    METRIC_TYPES = ["average_best_distance", "average_mean_distance", "average_generation"]

    file_paths = {
        "average_best_distance": {
            "homogeneity": "../glcm_bladder_average_gentl_results/40/average_best_absolute_distance_results/homogeneity_average_best_absolute_distance_results_40_rois.csv",
            "contrast": "../glcm_bladder_average_gentl_results/40/average_best_absolute_distance_results/contrast_average_best_absolute_distance_results_40_rois.csv",
            "correlation": "../glcm_bladder_average_gentl_results/40/average_best_absolute_distance_results/correlation_average_best_absolute_distance_results_40_rois.csv",
            "dissimilarity": "../glcm_bladder_average_gentl_results/40/average_best_absolute_distance_results/dissimilarity_average_best_absolute_distance_results_40_rois.csv",
            "energy": "../glcm_bladder_average_gentl_results/40/average_best_absolute_distance_results/energy_average_best_absolute_distance_results_40_rois.csv"
        },
        "average_mean_distance": {
            "homogeneity": "../glcm_bladder_average_gentl_results/40/average_mean_absolute_distance_results/homogeneity_average_mean_absolute_distance_results_40_rois.csv",
            "contrast": "../glcm_bladder_average_gentl_results/40/average_mean_absolute_distance_results/contrast_average_mean_absolute_distance_results_40_rois.csv",
            "correlation": "../glcm_bladder_average_gentl_results/40/average_mean_absolute_distance_results/correlation_average_mean_absolute_distance_results_40_rois.csv",
            "dissimilarity": "../glcm_bladder_average_gentl_results/40/average_mean_absolute_distance_results/dissimilarity_average_mean_absolute_distance_results_40_rois.csv",
            "energy": "../glcm_bladder_average_gentl_results/40/average_mean_absolute_distance_results/energy_average_mean_absolute_distance_results_40_rois.csv"
        },
        "average_generation": {
            "homogeneity": "../glcm_bladder_average_gentl_results/40/average_generation_absolute_distance_results/homogeneity_average_generation_absolute_distance_results_40_rois.csv",
            "contrast": "../glcm_bladder_average_gentl_results/40/average_generation_absolute_distance_results/contrast_average_generation_absolute_distance_results_40_rois.csv",
            "correlation": "../glcm_bladder_average_gentl_results/40/average_generation_absolute_distance_results/correlation_average_generation_absolute_distance_results_40_rois.csv",
            "dissimilarity": "../glcm_bladder_average_gentl_results/40/average_generation_absolute_distance_results/dissimilarity_average_generation_absolute_distance_results_40_rois.csv",
            "energy": "../glcm_bladder_average_gentl_results/40/average_generation_absolute_distance_results/energy_average_generation_absolute_distance_results_40_rois.csv"
        }
    }

    # Dictionary to store CV results for each metric type.
    cv_results = {metric: {ml: {} for ml in ML_METHODS} for metric in METRIC_TYPES}

    # Loop over metric types and GLCM features.
    for metric in METRIC_TYPES:
        for feature in GLCM_FEATURES:
            file_path = file_paths[metric][feature]
            print(f"\nProcessing file for {metric} - {feature}")
            merged_df, feature_column = process_file(file_path, metric, feature, cancer_grades_df)
            scores = run_classifiers_cv_all(merged_df, feature_column)
            for ml_method, score_dict in scores.items():
                cv_results[metric][ml_method][feature] = score_dict

    # Generate tables for each metric type and save as CSV and LaTeX.
    for metric in METRIC_TYPES:
        table_df = generate_table(cv_results, metric)
        csv_filename = f"cv_results_table_{metric}.csv"
        table_df.to_csv(csv_filename, index=False)
        latex_filename = f"cv_results_table_{metric}.tex"
        with open(latex_filename, "w") as f:
            f.write(table_df.to_latex(index=False))
