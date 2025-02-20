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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import class_weight

# Global variable for cancer grade order
cancer_grade_order = ['T0', 'Ta', 'Tis', 'T1', 'T2', 'T3', 'T4']

def evaluate_model_cv(model, X, y, model_name, feature_column, cv=10):
    """
    Evaluate a model using cross validation.
    Prints average F1 score (weighted) and its standard deviation,
    and returns a tuple (mean, std) (in percentage).
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    mean_score = np.mean(scores) * 100
    std_score = np.std(scores) * 100
    print(f"\nEvaluating model: {model_name}")
    print(f"F1-score (weighted) for {model_name} based on {feature_column}: {mean_score:.2f}% ± {std_score:.2f}%")
    return mean_score, std_score


# Each classifier function uses cross validation and returns (mean, std).
def decision_tree_cv(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=2, min_samples_leaf=7, random_state=42)
    return evaluate_model_cv(model, X, y, "Decision Tree", feature_column)


def knn_cv(merged_df, feature_column, n_neighbors=5):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return evaluate_model_cv(model, X, y, "KNN", feature_column)


def lda_cv(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = LinearDiscriminantAnalysis()
    return evaluate_model_cv(model, X, y, "LDA", feature_column)


def logistic_regression_cv(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = LogisticRegression(max_iter=1000, random_state=42)
    return evaluate_model_cv(model, X, y, "Logistic Regression", feature_column)


def random_forest_cv(merged_df, feature_column, n_estimators=100):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(cw))
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight_dict, random_state=42)
    return evaluate_model_cv(model, X, y, "Random Forest", feature_column)


def svm_cv(merged_df, feature_column):
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")
    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    model = SVC(kernel='linear', C=1.0, random_state=42)
    return evaluate_model_cv(model, X, y, "SVM", feature_column)


# Run all classifiers using cross validation and return a dictionary mapping ML method -> (mean, std)
def run_classifiers_cv(merged_df, feature_column):
    scores = {}
    scores["Decision Tree"] = decision_tree_cv(merged_df, feature_column)
    scores["KNN"] = knn_cv(merged_df, feature_column)
    scores["LDA"] = lda_cv(merged_df, feature_column)
    scores["Logistic Regression"] = logistic_regression_cv(merged_df, feature_column)
    scores["Random Forest"] = random_forest_cv(merged_df, feature_column)
    scores["SVM"] = svm_cv(merged_df, feature_column)
    return scores


# Process a single file: load, merge with cancer grades, set ordering, and determine feature column.
def process_file(file_path, metric_type, glcm_feature, cancer_grades_df):
    """
    Load the average file (e.g., average_best_distance, average_mean_distance, or average_generation),
    merge with cancer grades, and return the merged dataframe along with the feature column name.
    """
    avg_df = pd.read_csv(file_path)
    merged_df = avg_df.merge(cancer_grades_df, on='patient_id', how='left')
    # Set cancer grade order for proper sorting/plotting
    merged_df['cancer_grade'] = pd.Categorical(merged_df['cancer_grade'], categories=cancer_grade_order, ordered=True)
    # Map the expected metric_type to the actual column in the CSV:
    if metric_type == "average_generation":
        feature_column = "average_generation"
    elif metric_type in ["average_best_distance", "average_mean_distance"]:
        # Both types use 'average_distance' column in CSV
        if "average_distance" in merged_df.columns:
            feature_column = "average_distance"
        else:
            raise ValueError(f"Column 'average_distance' not found in {file_path}.")
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    return merged_df, feature_column


# Plot the grouped bar chart with error bars for a given metric type.
def plot_results(results, metric_type):
    """
    Generate a grouped bar chart with error bars for CV F1 scores and save as PDF.

    - X-axis: ML methods.
    - Each ML method group contains bars for each of the five GLCM features.
    - Y-axis: F1-scores(%) (with error bars showing standard deviation).
    - The chart has thick borders, major horizontal grid lines at 20, 40, 60, 80 in grey (drawn behind the bars),
      and minor grid lines (every 10) in light grey.
    - Each bar is annotated with its value (mean) rotated vertically.
    - The legend is placed at the top (outside the plot) in a horizontal layout.
    - A bold title is placed at the bottom.
    - The plot is saved as a PDF file.

    :param results: Dictionary with structure results[metric_type][ml_method][glcm_feature] = (mean, std)
    :param metric_type: Metric type string (e.g., "average_best_distance") used for labeling.
    """
    # Map metric types to descriptive labels.
    metric_labels = {
        "average_best_distance": "average best distance",
        "average_mean_distance": "average mean distance",
        "average_generation": "average generation"
    }
    metric_desc = metric_labels.get(metric_type, metric_type)

    # Get ML methods and GLCM features.
    ml_methods = list(results[metric_type].keys())
    glcm_features = list(next(iter(results[metric_type].values())).keys())
    n_methods = len(ml_methods)
    n_features = len(glcm_features)
    x = np.arange(n_methods)
    width = 0.15  # width for each bar

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars for each GLCM feature with error bars and annotate each bar.
    for i, feature in enumerate(glcm_features):
        positions = x + i * width - (n_features - 1) * width / 2
        means = [results[metric_type][ml][feature][0] for ml in ml_methods]
        stds = [results[metric_type][ml][feature][1] for ml in ml_methods]
        bars = ax.bar(positions, means, width, yerr=stds, capsize=5, label=feature)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, rotation=90)

    # Set x-axis ticks and labels (displayed horizontally).
    ax.set_xticks(x)
    ax.set_xticklabels(ml_methods, rotation=0)
    ax.set_ylabel("F1-scores(%)")

    # Configure grid: major ticks at 20, 40, 60, 80; minor ticks every 10.
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticks(np.arange(0, 101, 10), minor=True)
    ax.set_axisbelow(True)  # Draw grid behind bars.
    ax.grid(which='major', axis='y', color='grey', linewidth=1)
    ax.grid(which='minor', axis='y', linestyle=':', color='lightgrey', linewidth=0.5)

    # Thicken the plot border (spines).
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Place legend at the top outside the plot; adjust bbox to reduce top blank space.
    ax.legend(title="GLCM Feature", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=n_features)

    # Place a bold title at the bottom; adjust y-coordinate to reduce distance to x-axis labels.
    fig.text(0.5, 0.03,
             f"F1 Scores (%) for Classifying MIBC vs NMIBC using five GLCM Features ({metric_desc})",
             ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Adjust layout to reduce top and bottom blank spaces.
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    # Save the figure as a PDF file.
    pdf_filename = f"barplot_10_fold_cv_{metric_type}.pdf"
    fig.savefig(pdf_filename)
    plt.show()


# ===================== Main =====================
if __name__ == "__main__":
    # Step 1: load cancer grades file
    cancer_grades_df = pd.read_csv(
        '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv')
    cancer_grades_df.rename(columns={'Anonymized ID': 'patient_id'}, inplace=True)
    cancer_grades_df.rename(columns={'Final Path': 'cancer_grade'}, inplace=True)
    cancer_grades_df.dropna(subset=['patient_id', 'cancer_grade'], inplace=True)
    cancer_grades_df = cancer_grades_df[['patient_id', 'cancer_grade']]

    # Define the ML methods (for ordering in the plot)
    ML_METHODS = ["Decision Tree", "KNN", "LDA", "Logistic Regression", "Random Forest", "SVM"]
    # Define the 5 GLCM features (names as used in labeling)
    GLCM_FEATURES = ["homogeneity", "contrast", "correlation", "dissimilarity", "energy"]
    # Define the three metric types
    METRIC_TYPES = ["average_best_distance", "average_mean_distance", "average_generation"]

    # Define a dictionary holding file paths.
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
    # results[metric_type][ml_method][glcm_feature] = (mean, std)
    results = {metric: {ml: {} for ml in ML_METHODS} for metric in METRIC_TYPES}

    # Loop over metric types and GLCM features (15 files in total)
    for metric in METRIC_TYPES:
        for feature in GLCM_FEATURES:
            file_path = file_paths[metric][feature]
            print(f"\nProcessing file for {metric} - {feature}")
            merged_df, feature_column = process_file(file_path, metric, feature, cancer_grades_df)
            # Run all classifiers using cross validation and get (mean, std)
            scores = run_classifiers_cv(merged_df, feature_column)
            for ml_method, score_tuple in scores.items():
                results[metric][ml_method][feature] = score_tuple

    # Generate a grouped bar plot (with error bars) for each metric type.
    for metric in METRIC_TYPES:
        plot_results(results, metric)
