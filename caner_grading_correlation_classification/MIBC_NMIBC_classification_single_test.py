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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import class_weight

# Global variable for cancer grade order
cancer_grade_order = ['T0', 'Ta', 'Tis', 'T1', 'T2', 'T3', 'T4']


# ---------------------------- Model Evaluation ----------------------------
def evaluate_model(y_true, y_pred, model_name, feature_column):
    """
    Evaluate the model by printing accuracy and weighted F1 score.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param model_name: Name of the model.
    :param feature_column: Feature column used in the model.
    :return: F1 score (weighted) in percentage.
    """
    print(f"\nEvaluating model: {model_name}")
    accuracy_val = accuracy_score(y_true, y_pred) * 100
    f1_score_val = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    print(f"Accuracy for MIBC vs NMIBC based on {feature_column}: {accuracy_val:.2f}%")
    print(f"F1-score for MIBC vs NMIBC based on {feature_column}: {f1_score_val:.2f}%")
    print(classification_report(y_true, y_pred, zero_division=0))
    return f1_score_val


# ---------------------------- Classifier Functions ----------------------------
def decision_tree(merged_df, feature_column):
    """
    Train and evaluate a Decision Tree classifier.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(
        class_weight='balanced', criterion='entropy',
        max_depth=2, min_samples_leaf=7, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "Decision Tree", feature_column)


def knn_classifier(merged_df, feature_column, n_neighbors=5):
    """
    Train and evaluate a K-Nearest Neighbors classifier.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "KNN", feature_column)


def lda_classifier(merged_df, feature_column):
    """
    Train and evaluate a Linear Discriminant Analysis classifier.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "LDA", feature_column)


def logistic_regression(merged_df, feature_column):
    """
    Train and evaluate a Logistic Regression classifier.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "Logistic Regression", feature_column)


def random_forest(merged_df, feature_column, n_estimators=100):
    """
    Train and evaluate a Random Forest classifier.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight_dict, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "Random Forest", feature_column)


def svm_classifier(merged_df, feature_column):
    """
    Train and evaluate a Support Vector Machine (SVM) classifier.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred, "SVM", feature_column)


def run_classifiers(merged_df, feature_column):
    """
    Run all classifiers and return a dictionary mapping ML method to F1 score.
    """
    scores = {}
    scores["Decision Tree"] = decision_tree(merged_df, feature_column)
    scores["KNN"] = knn_classifier(merged_df, feature_column)
    scores["LDA"] = lda_classifier(merged_df, feature_column)
    scores["Logistic Regression"] = logistic_regression(merged_df, feature_column)
    scores["Random Forest"] = random_forest(merged_df, feature_column)
    scores["SVM"] = svm_classifier(merged_df, feature_column)
    return scores


# ---------------------------- Data Processing ----------------------------
def process_file(file_path, metric_type, glcm_feature, cancer_grades_df):
    """
    Load the GLCM results file, merge it with cancer grades, and return the merged dataframe and feature column.

    :param file_path: Path to the CSV file containing GLCM results.
    :param metric_type: Expected metric type ("average_best_distance", "average_mean_distance", or "average_generation").
    :param glcm_feature: Name of the GLCM feature (for labeling purposes).
    :param cancer_grades_df: DataFrame containing cancer grades.
    :return: merged_df and feature_column (the column name used for classification).
    """
    avg_df = pd.read_csv(file_path)
    merged_df = avg_df.merge(cancer_grades_df, on='patient_id', how='left')
    # Set the order for cancer_grade for proper sorting and plotting
    merged_df['cancer_grade'] = pd.Categorical(merged_df['cancer_grade'], categories=cancer_grade_order, ordered=True)

    # Map metric_type to the actual column name in the CSV.
    if metric_type == "average_generation":
        feature_column = "average_generation"
    elif metric_type in ["average_best_distance", "average_mean_distance"]:
        # Both types use the actual column name "average_distance" in the CSV
        if "average_distance" in merged_df.columns:
            feature_column = "average_distance"
        else:
            raise ValueError(f"Column 'average_distance' not found in {file_path}.")
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    return merged_df, feature_column


# ---------------------------- Plotting Function ----------------------------
def plot_results(results, metric_type):
    """
    Generate a grouped bar chart for F1 scores and save as PDF.

    - X-axis: ML methods.
    - Each ML method group contains bars for each of the five GLCM features.
    - Y-axis: F1 score (%), labeled as "F1-scores(%)".
    - Styling:
         * Plot border (spines) are thickened.
         * Major horizontal grid lines at 20, 40, 60, 80 are grey and drawn behind the bars.
         * Minor horizontal grid lines (every 10) are lighter and thinner.
         * Bar value annotations are rotated vertically.
         * Legend is placed at the top (outside the plot) in a horizontal layout for all metric types.
         * The title is placed at the bottom in bold.
    - The plot is saved as a PDF file.

    :param results: Dictionary with structure results[metric_type][ml_method][glcm_feature] = F1 score.
    :param metric_type: Metric type string (e.g., "average_best_distance") used for labeling.
    """
    # Mapping metric types to descriptive labels
    metric_labels = {
        "average_best_distance": "average best distance",
        "average_mean_distance": "average mean distance",
        "average_generation": "average generation"
    }
    metric_desc = metric_labels.get(metric_type, metric_type)

    # Get ML methods and GLCM features
    ml_methods = list(results[metric_type].keys())
    glcm_features = list(next(iter(results[metric_type].values())).keys())

    n_methods = len(ml_methods)
    n_features = len(glcm_features)
    x = np.arange(n_methods)
    width = 0.15  # width of each bar

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars for each GLCM feature and annotate values rotated vertically
    for i, feature in enumerate(glcm_features):
        positions = x + i * width - (n_features - 1) * width / 2
        scores = [results[metric_type][ml_method][feature] for ml_method in ml_methods]
        bars = ax.bar(positions, scores, width, label=feature)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, rotation=90)

    # Set x-axis ticks and labels horizontally
    ax.set_xticks(x)
    ax.set_xticklabels(ml_methods, rotation=0)
    ax.set_ylabel("F1-scores(%)")

    # Configure grid lines: major ticks at 20, 40, 60, 80; minor ticks every 10.
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticks(np.arange(0, 101, 10), minor=True)
    ax.set_axisbelow(True)  # Draw grid below bars
    ax.grid(which='major', axis='y', color='grey', linewidth=1)
    ax.grid(which='minor', axis='y', linestyle=':', color='lightgrey', linewidth=0.5)

    # Thicken the border (spines)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Place legend at the top outside the plot for all metric types
    ax.legend(title="GLCM Feature", loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=n_features)

    # Instead of a top title, place a bold title at the bottom.
    fig.text(0.5, 0.03,
             f"F1 scores (%) for classifying MIBC vs NMIBC using five GLCM features ({metric_desc})",
             ha="center", va="bottom", fontsize=14, fontweight="bold")

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust layout to make room for bottom title

    # Save the figure as a PDF file
    pdf_filename = f"barplot_{metric_type}.pdf"
    fig.savefig(pdf_filename)
    plt.show()


# ---------------------------- Main Script ----------------------------
if __name__ == "__main__":
    # Step 1: Load the cancer grades file
    cancer_grades_df = pd.read_csv(
        '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv')
    cancer_grades_df.rename(columns={'Anonymized ID': 'patient_id', 'Final Path': 'cancer_grade'}, inplace=True)
    cancer_grades_df.dropna(subset=['patient_id', 'cancer_grade'], inplace=True)
    cancer_grades_df = cancer_grades_df[['patient_id', 'cancer_grade']]

    # Define the ML methods (for ordering in the plot)
    ML_METHODS = ["Decision Tree", "KNN", "LDA", "Logistic Regression", "Random Forest", "SVM"]
    # Define the 5 GLCM features (used for labeling)
    GLCM_FEATURES = ["homogeneity", "contrast", "correlation", "dissimilarity", "energy"]
    # Define the three metric types
    METRIC_TYPES = ["average_best_distance", "average_mean_distance", "average_generation"]

    # Dictionary holding file paths:
    # Structure: file_paths[metric_type][glcm_feature] = file_path
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

    # Dictionary to store results:
    # results[metric_type][ml_method][glcm_feature] = F1 score
    results = {metric: {ml: {} for ml in ML_METHODS} for metric in METRIC_TYPES}

    # Loop over metric types and GLCM features (total 15 files)
    for metric in METRIC_TYPES:
        for feature in GLCM_FEATURES:
            file_path = file_paths[metric][feature]
            print(f"\nProcessing file for {metric} - {feature}")
            # Process the file: load data and merge with cancer grades
            merged_df, feature_column = process_file(file_path, metric, feature, cancer_grades_df)
            # Run all classifiers and obtain F1 scores (each classifier prints its own results)
            scores = run_classifiers(merged_df, feature_column)
            # Save F1 scores in the results dictionary
            for ml_method, f1 in scores.items():
                results[metric][ml_method][feature] = f1

    # Generate a grouped bar plot for each metric type and save as PDF.
    for metric in METRIC_TYPES:
        plot_results(results, metric)



